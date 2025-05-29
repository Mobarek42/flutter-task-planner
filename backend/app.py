import sqlite3
import datetime
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import random
import math
import threading
import uuid
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin
DATABASE = 'tasks.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, name TEXT, description TEXT, deadline TEXT, duration INTEGER,
            priority INTEGER, dependencies TEXT, resources TEXT, start_time TEXT)''')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'start_time' not in columns:
            conn.execute('ALTER TABLE tasks ADD COLUMN start_time TEXT')
            logger.info("Column 'start_time' added to table 'tasks'")
        conn.execute('''CREATE TABLE IF NOT EXISTS resources (
            id TEXT PRIMARY KEY, name TEXT, type TEXT, is_available INTEGER, cost REAL)''')

init_db()

@app.route('/tasks', methods=['GET'])
def get_tasks():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        tasks = conn.execute('SELECT * FROM tasks').fetchall()
    return jsonify([{
        **dict(task),
        'dependencies': task['dependencies'].split(',') if task['dependencies'] else [],
        'resources': json.loads(task['resources']) if task['resources'] else []
    } for task in tasks])

@app.route('/tasks', methods=['POST'])
def add_task():
    task = request.get_json()
    logger.info(f"Task received: {task}")
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('INSERT INTO tasks (id, name, description, deadline, duration, priority, dependencies, resources, start_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (task['id'], task['name'], task['description'], task['deadline'], task['duration'],
                      task['priority'], ','.join(task['dependencies']), json.dumps(task['resources']), task.get('startTime')))
    return '', 201

@app.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    return '', 200

@app.route('/resources', methods=['GET'])
def get_resources():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        resources = conn.execute('SELECT * FROM resources').fetchall()
    return jsonify([dict(resource) for resource in resources])

@app.route('/resources', methods=['POST'])
def add_resource():
    resource = request.get_json()
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('INSERT INTO resources VALUES (?, ?, ?, ?, ?)',
                     (resource['id'], resource['name'], resource['type'], 1 if resource['isAvailable'] else 0, resource['cost']))
    return '', 201

@app.route('/resources/<resource_id>', methods=['DELETE'])
def delete_resource(resource_id):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('DELETE FROM resources WHERE id = ?', (resource_id,))
    return '', 200

def is_task_user_fixed(task):
    """Détermine si une tâche a un startTime fixe défini EXPLICITEMENT par l'utilisateur"""
    # Une tâche est considérée comme fixée par l'utilisateur seulement si :
    # 1. Elle a un startTime
    # 2. Ce startTime n'a PAS été calculé par un algorithme (_computed_start_time = False ou absent)
    return ('startTime' in task and task['startTime'] and 
            not task.get('_computed_start_time', False))

def is_task_fixed(task):
    """Détermine si une tâche a un startTime fixe (défini par l'utilisateur)"""
    return is_task_user_fixed(task)

def get_initial_solution(tasks, resources):
    now = datetime.datetime.now()
    start_times = {}
    
    # Sort tasks by priority and deadline
    sorted_tasks = sorted(tasks, key=lambda t: (
        -t.get('priority', 0),
        datetime.datetime.fromisoformat(t['deadline'].replace('Z', '+00:00'))
    ))
    
    for task in sorted_tasks:
        if is_task_fixed(task):
            start_times[task['id']] = (datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00')) - now).total_seconds() / 3600
        else:
            earliest_start = 0
            
            # Respect dependencies
            dependencies = task.get('dependencies', [])
            for dep_id in dependencies:
                if dep_id and dep_id in start_times:
                    dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                    if dep_task:
                        dep_end = start_times[dep_id] + dep_task['duration']
                        earliest_start = max(earliest_start, dep_end)
            
            # Avoid resource conflicts
            task_resources = task.get('resources', [])
            for res_id in task_resources:
                for other_task in tasks:
                    if other_task['id'] != task['id'] and res_id in other_task.get('resources', []) and other_task['id'] in start_times:
                        other_start = start_times[other_task['id']]
                        other_end = other_start + other_task['duration']
                        if earliest_start < other_end:
                            earliest_start = other_end
            
            start_times[task['id']] = earliest_start
    
    return start_times

def evaluate_solution(start_times, tasks, resources, now=None):
    """
    Evaluate the quality of a solution (makespan).
    CORRECTION: Calcule correctement le makespan en tenant compte des tâches fixes
    """
    if now is None:
        now = datetime.datetime.now()
    
    makespan = 0
    
    for task in tasks:
        task_id = task['id']
        
        if is_task_fixed(task):
            # Pour les tâches fixes, calculer le temps de fin à partir du startTime absolu
            start_time = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
            end_time = start_time + datetime.timedelta(hours=task['duration'])
            # Convertir en heures relatives au moment actuel
            task_end_hours = (end_time - now).total_seconds() / 3600
            makespan = max(makespan, task_end_hours)
        elif task_id in start_times:
            # Pour les tâches optimisables, utiliser les temps relatifs
            start = start_times[task_id]
            end = start + task['duration']
            makespan = max(makespan, end)
    
    return makespan

def strictly_enforce_dependencies(solution, tasks):
    """
    Ensure that dependency constraints are strictly enforced.
    Modifies the solution in place.
    
    Args:
        solution: Dictionary mapping task IDs to start times (en heures relatives)
        tasks: List of task objects
    """
    now = datetime.datetime.now()
    
    # Create a dictionary for quick task lookup
    task_dict = {task['id']: task for task in tasks}
    
    # Build a dependency graph
    dep_graph = {}
    for task in tasks:
        task_id = task['id']
        dep_graph[task_id] = task.get('dependencies', [])
    
    # Topologically sort tasks (respecting dependencies)
    visited = set()
    temp_visited = set()
    order = []
    
    def visit(task_id):
        if task_id in temp_visited:  # This means we have a cycle
            return
        if task_id in visited:
            return
        
        temp_visited.add(task_id)
        for dep_id in dep_graph.get(task_id, []):
            if dep_id:  # Skip empty dependencies
                visit(dep_id)
        
        temp_visited.remove(task_id)
        visited.add(task_id)
        order.append(task_id)
    
    for task_id in dep_graph:
        if task_id not in visited:
            visit(task_id)
    
    # Process tasks in topological order
    for task_id in reversed(order):
        if task_id in solution:
            # Get the task's start time
            start_time = solution[task_id]
            
            # Check all dependencies
            dependencies = dep_graph.get(task_id, [])
            for dep_id in dependencies:
                if dep_id:
                    dep_task = task_dict[dep_id]
                    
                    if is_task_fixed(dep_task):
                        # CORRECTION: Pour les dépendances fixes, calculer le temps de fin en heures relatives
                        dep_start_abs = datetime.datetime.fromisoformat(dep_task['startTime'].replace('Z', '+00:00'))
                        dep_end_abs = dep_start_abs + datetime.timedelta(hours=dep_task['duration'])
                        dep_end_relative = (dep_end_abs - now).total_seconds() / 3600
                        
                        if start_time < dep_end_relative:
                            solution[task_id] = dep_end_relative
                    elif dep_id in solution:
                        # Pour les dépendances optimisables
                        dep_duration = dep_task['duration']
                        dep_end_time = solution[dep_id] + dep_duration
                        
                        if start_time < dep_end_time:
                            solution[task_id] = dep_end_time

def repair_solution(solution, tasks):
    """
    Repair a solution to respect dependency constraints and resource constraints.
    This is a more thorough implementation that combines dependency and resource checks.
    """
    # First, strictly enforce dependencies
    strictly_enforce_dependencies(solution, tasks)
    
    # Create a dictionary for quick task lookup
    task_dict = {task['id']: task for task in tasks}
    
    # Resolve resource conflicts
    has_conflicts = True
    max_iterations = 100  # Prevent infinite loops
    iteration = 0
    
    while has_conflicts and iteration < max_iterations:
        has_conflicts = False
        iteration += 1
        
        # Check for resource conflicts
        for i, task1 in enumerate(tasks):
            task1_id = task1['id']
            if task1_id not in solution:
                continue
                
            task1_start = solution[task1_id]
            task1_end = task1_start + task1['duration']
            task1_resources = task1.get('resources', [])
            
            for j, task2 in enumerate(tasks):
                if i == j:
                    continue
                    
                task2_id = task2['id']
                if task2_id not in solution:
                    continue
                    
                task2_start = solution[task2_id]
                task2_end = task2_start + task2['duration']
                task2_resources = task2.get('resources', [])
                
                # Check if tasks overlap in time
                if not (task1_end <= task2_start or task2_end <= task1_start):
                    # Check if tasks share any resources
                    shared_resources = set(task1_resources).intersection(set(task2_resources))
                    if shared_resources:
                        has_conflicts = True
                        
                        # Decide which task to move
                        # Prefer to move the task with lower priority or later deadline
                        task1_priority = task1.get('priority', 0)
                        task2_priority = task2.get('priority', 0)
                        
                        if task1_priority > task2_priority:
                            # Move task2 after task1
                            solution[task2_id] = task1_end
                        elif task2_priority > task1_priority:
                            # Move task1 after task2
                            solution[task1_id] = task2_end
                        else:
                            # If priorities are equal, move the task with a later deadline
                            task1_deadline = datetime.datetime.fromisoformat(task1['deadline'].replace('Z', '+00:00'))
                            task2_deadline = datetime.datetime.fromisoformat(task2['deadline'].replace('Z', '+00:00'))
                            
                            if task1_deadline > task2_deadline:
                                solution[task1_id] = task2_end
                            else:
                                solution[task2_id] = task1_end
        
        # After resolving resource conflicts, re-enforce dependencies
        strictly_enforce_dependencies(solution, tasks)
    
    return solution

def simulated_annealing(tasks, resources, initial_solution=None, params=None):
    """
    Optimize the schedule using simulated annealing with strict dependency constraints.
    
    Args:
        tasks: List of tasks
        resources: List of resources
        initial_solution: Initial solution (optional)
        params: Additional parameters (optional)
        
    Returns:
        Best solution found
    """
    # Parameters for simulated annealing
    params = params or {}
    initial_temp = params.get('initial_temp', 1000.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    max_iterations = params.get('max_iterations', 10000)
    no_improvement_limit = params.get('no_improvement_limit', 1000)
    
    # CORRECTION: Passer le moment actuel pour cohérence
    now = datetime.datetime.now()
    
    # Create a task dictionary for quick lookups
    task_dict = {task['id']: task for task in tasks}
    
    # Initialize solution
    current_solution = initial_solution or get_initial_solution(tasks, resources)
    
    # Apply strict dependency enforcement to initial solution
    strictly_enforce_dependencies(current_solution, tasks)
    
    current_cost = evaluate_solution(current_solution, tasks, resources, now)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    # Simulated annealing algorithm
    temp = initial_temp
    iteration = 0
    no_improvement = 0
    
    while temp > 1 and iteration < max_iterations and no_improvement < no_improvement_limit:
        new_solution = current_solution.copy()
        
        # Select a task to modify (only non-fixed tasks)
        task_ids = [t['id'] for t in tasks if not is_task_fixed(t)]
        if not task_ids:
            break
        task_id = random.choice(task_ids)
        
        # Apply perturbation
        perturbation = random.uniform(-2 * (temp / initial_temp), 2 * (temp / initial_temp))
        new_solution[task_id] += perturbation
        
        # Enforce non-negative start times
        new_solution[task_id] = max(0, new_solution[task_id])
        
        # Strictly enforce dependency constraints
        strictly_enforce_dependencies(new_solution, tasks)
        
        # Repair solution to handle resource conflicts
        repair_solution(new_solution, tasks)
        
        # Evaluate new solution
        new_cost = evaluate_solution(new_solution, tasks, resources, now)
        
        # Metropolis criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = new_solution
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        # Cooling
        temp *= cooling_rate
        iteration += 1
    
    # Final check to ensure all dependencies are still satisfied
    strictly_enforce_dependencies(best_solution, tasks)
    
    # CORRECTION: Recalculer le coût final pour s'assurer de la cohérence
    final_cost = evaluate_solution(best_solution, tasks, resources, now)
    
    return best_solution, final_cost, iteration, temp

results = {}

@app.route('/optimize', methods=['POST'])
def optimize_schedule_async():
    data = request.get_json()
    # CORRECTION PRINCIPALE : Faire une copie profonde des tâches pour éviter les modifications
    tasks = copy.deepcopy(data['tasks'])
    resources = data['resources']
    method = data.get('method', 'PuLP')  # Default to PuLP
    params = data.get('params', {})  # Additional parameters
    
    # CORRECTION CLÉE : Si pas de tâches vraiment fixes par l'utilisateur, permettre la réoptimisation
    user_fixed_tasks = [t for t in tasks if is_task_user_fixed(t)]
    if len(user_fixed_tasks) == 0:
        logger.info("No user-fixed tasks found, clearing all computed start times for reoptimization")
        for task in tasks:
            if 'startTime' in task and task.get('_computed_start_time', False):
                del task['startTime']
            if '_computed_start_time' in task:
                del task['_computed_start_time']
    
    result_id = str(uuid.uuid4())
    results[result_id] = {"status": "pending"}

    def optimize_schedule(tasks, resources, method, params, result_id):
        try:
            now = datetime.datetime.now()
            # CORRECTION : Utiliser la nouvelle fonction is_task_fixed
            fixed_tasks = [t for t in tasks if is_task_fixed(t)]
            tasks_to_optimize = [t for t in tasks if not is_task_fixed(t)]
            
            logger.info(f"Fixed tasks: {len(fixed_tasks)}, Tasks to optimize: {len(tasks_to_optimize)} for method {method}")

            if method == 'PuLP':
                # CORRECTION : Vérifier qu'il y a des tâches à optimiser
                if len(tasks_to_optimize) == 0:
                    logger.warning(f"No tasks to optimize for PuLP method (result_id: {result_id})")
                    # Calculer le makespan avec les tâches fixes existantes
                    makespan_val = 0
                    final_tasks = []
                    for task in tasks:
                        if is_task_fixed(task):
                            start_time = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
                            end_time = start_time + datetime.timedelta(hours=task['duration'])
                            task_end_hours = (end_time - now).total_seconds() / 3600
                            makespan_val = max(makespan_val, task_end_hours)
                        final_tasks.append(task)
                    
                    throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                    
                    results[result_id] = {
                        "status": "completed",
                        "tasks": final_tasks,
                        "metrics": {
                            "makespan": makespan_val,
                            "throughput": throughput,
                            "resource_utilization": {},
                            "penalties": {
                                "deadline_violations": 0.0,
                                "dependency_violations": 0.0,
                                "resource_conflicts": 0.0
                            }
                        }
                    }
                    return

                prob = LpProblem("Task_Scheduling", LpMinimize)
                start_times = {task['id']: LpVariable(f"start_{task['id']}", 0) for task in tasks_to_optimize}
                makespan = LpVariable("makespan", 0)

                # Makespan constraint
                for task in tasks_to_optimize:
                    prob += makespan >= start_times[task['id']] + task['duration']
                
                # Contraintes pour les tâches fixes
                for task in fixed_tasks:
                    start_time = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
                    end_time = start_time + datetime.timedelta(hours=task['duration'])
                    task_end_hours = (end_time - now).total_seconds() / 3600
                    prob += makespan >= task_end_hours
                
                # Dependency constraints
                for task in tasks_to_optimize:
                    dependencies = task.get('dependencies', [])
                    for dep_id in dependencies:
                        if dep_id:
                            dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                            if dep_task:
                                if is_task_fixed(dep_task):
                                    dep_start_str = dep_task['startTime'].replace('Z', '')
                                    if '+' in dep_start_str:
                                        dep_start_str = dep_start_str.split('+')[0]
                                    dep_start = datetime.datetime.fromisoformat(dep_start_str)
                                    dep_end = (dep_start - now).total_seconds() / 3600 + dep_task['duration']
                                    prob += start_times[task['id']] >= dep_end
                                else:
                                    prob += start_times[task['id']] >= start_times[dep_id] + dep_task['duration']

                # Resource constraints
                resource_usage = {}
                for task in tasks_to_optimize:
                    resources_list = task.get('resources', [])
                    for res_id in resources_list:
                        resource_usage.setdefault(res_id, []).append(task)

                for res_id, res_tasks in resource_usage.items():
                    for i in range(len(res_tasks)):
                        for j in range(i + 1, len(res_tasks)):
                            t1, t2 = res_tasks[i], res_tasks[j]
                            b = LpVariable(f"b_{t1['id']}_{t2['id']}", cat='Binary')
                            prob += start_times[t1['id']] + t1['duration'] <= start_times[t2['id']] + 1000 * (1 - b)
                            prob += start_times[t2['id']] + t2['duration'] <= start_times[t1['id']] + 1000 * b

                # Deadline constraints (optional)
                if params.get('enforce_deadlines', False):
                    for task in tasks_to_optimize:
                        deadline_str = task['deadline'].replace('Z', '')
                        if '+' in deadline_str:
                            deadline_str = deadline_str.split('+')[0]
                        deadline = datetime.datetime.fromisoformat(deadline_str)
                        deadline_hours = (deadline - now).total_seconds() / 3600
                        prob += start_times[task['id']] + task['duration'] <= deadline_hours

                # Objective function: minimize makespan
                prob += makespan

                prob.solve()
                if LpStatus[prob.status] == 'Optimal':
                    logger.info(f"Optimal solution found with PuLP for result_id {result_id}")
                    optimized_start_times = {task['id']: value(start_times[task['id']]) for task in tasks_to_optimize}
                    
                    makespan_val = value(makespan)
                    throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                    logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id}")
                    
                    # Calculate additional metrics
                    metrics = {
                        "makespan": makespan_val,
                        "throughput": throughput,
                    }
                    
                    # Calculate penalties
                    penalties = {
                        "deadline_violations": 0.0,
                        "dependency_violations": 0.0,
                        "resource_conflicts": 0.0
                    }
                    
                    # Update tasks with start times
                    final_tasks = []
                    for task in tasks:
                        if task['id'] in optimized_start_times:
                            start_hour = optimized_start_times[task['id']]
                            start_time = now + datetime.timedelta(hours=start_hour)
                            task['startTime'] = start_time.isoformat()
                            # CORRECTION : Marquer les startTime calculés
                            task['_computed_start_time'] = True
                            end = start_time + datetime.timedelta(hours=task['duration'])
                            deadline_str = task['deadline'].replace('Z', '')
                            if '+' in deadline_str:
                                deadline_str = deadline_str.split('+')[0]
                            deadline = datetime.datetime.fromisoformat(deadline_str)
                            if end > deadline:
                                penalties["deadline_violations"] += (end - deadline).total_seconds() / 3600
                        final_tasks.append(task)
                    
                    # Calculate resource utilization
                    resource_utilization = {}
                    for res_id in set(res_id for task in tasks for res_id in task.get('resources', [])):
                        total_time = 0
                        for task in final_tasks:
                            if res_id in task.get('resources', []) and 'startTime' in task and task['startTime']:
                                total_time += task['duration']
                        resource_utilization[res_id] = total_time / makespan_val if makespan_val > 0 else 0
                    
                    metrics["resource_utilization"] = resource_utilization
                    metrics["penalties"] = penalties
                    
                    results[result_id] = {
                        "status": "completed",
                        "tasks": final_tasks,
                        "metrics": metrics
                    }
                    logger.info(f"Stored result for {result_id}: {results[result_id]}")
                else:
                    logger.error(f"PuLP failed for result_id {result_id} (status: {LpStatus[prob.status]})")
                    results[result_id] = {
                        "status": "failed",
                        "error": f"PuLP failed (status: {LpStatus[prob.status]})"
                    }
                    return

            elif method == 'SimulatedAnnealing':
                sa_params = {
                    'initial_temp': params.get('initial_temp', 1000.0),
                    'cooling_rate': params.get('cooling_rate', 0.95),
                    'max_iterations': params.get('max_iterations', 10000),
                    'no_improvement_limit': params.get('no_improvement_limit', 1000)
                }
                initial_solution = get_initial_solution(tasks, resources)
                # Ensure initial solution respects dependencies
                strictly_enforce_dependencies(initial_solution, tasks)
                
                best_solution, best_cost, iterations, final_temp = simulated_annealing(
                    tasks, resources, initial_solution=initial_solution, params=sa_params
                )
                
                makespan_val = best_cost
                throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id} using SA")
                
                # Calculate additional metrics
                metrics = {
                    "makespan": makespan_val,
                    "throughput": throughput,
                    "iterations": iterations,
                    "final_temperature": final_temp
                }
                
                # Calculate penalties
                penalties = {
                    "deadline_violations": 0.0,
                    "dependency_violations": 0.0,
                    "resource_conflicts": 0.0
                }
                
                # Update tasks with start times
                final_tasks = []
                for task in tasks:
                    if task['id'] in best_solution:
                        start_hour = best_solution[task['id']]
                        start_time = now + datetime.timedelta(hours=start_hour)
                        task['startTime'] = start_time.isoformat()
                        # CORRECTION : Marquer les startTime calculés
                        task['_computed_start_time'] = True
                        end = start_time + datetime.timedelta(hours=task['duration'])
                        deadline_str = task['deadline'].replace('Z', '')
                        if '+' in deadline_str:
                            deadline_str = deadline_str.split('+')[0]
                        deadline = datetime.datetime.fromisoformat(deadline_str)
                        if end > deadline:
                            penalties["deadline_violations"] += (end - deadline).total_seconds() / 3600
                    final_tasks.append(task)
                
                # Validate dependency constraints
                for task in final_tasks:
                    dependencies = task.get('dependencies', [])
                    for dep_id in dependencies:
                        if dep_id:
                            dep_task = next((t for t in final_tasks if t['id'] == dep_id), None)
                            if (
                                dep_task and 'startTime' in dep_task and 'startTime' in task and 
                                datetime.datetime.fromisoformat(task['startTime'].replace('Z', '')) < 
                                datetime.datetime.fromisoformat(dep_task['startTime'].replace('Z', '')) + 
                                datetime.timedelta(hours=dep_task['duration'])
                            ):
                                penalties["dependency_violations"] += 1
                
                # Calculate resource utilization
                resource_utilization = {}
                for res_id in set(res_id for task in tasks for res_id in task.get('resources', [])):
                    total_time = 0
                    for task in final_tasks:
                        if res_id in task.get('resources', []) and 'startTime' in task and task['startTime']:
                            total_time += task['duration']
                    resource_utilization[res_id] = total_time / makespan_val if makespan_val > 0 else 0
                
                metrics["resource_utilization"] = resource_utilization
                metrics["penalties"] = penalties
                
                results[result_id] = {
                    "status": "completed",
                    "tasks": final_tasks,
                    "metrics": metrics
                }
                logger.info(f"Stored result for {result_id}: {results[result_id]}")
            else:
                logger.error(f"Invalid optimization method '{method}' for result_id {result_id}")
                results[result_id] = {"status": "failed", "error": "Invalid optimization method"}
        except Exception as e:
            logger.exception(f"Optimization failed for result_id {result_id}: {e}")
            results[result_id] = {"status": "failed", "error": str(e)}

    thread = threading.Thread(target=optimize_schedule, args=(tasks, resources, method, params, result_id))
    thread.start()

    # Return 202 Accepted status code for asynchronous operation
    return jsonify({"status": "pending", "id": result_id}), 202

@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id):
    result = results.get(result_id, {"status": "not_found"})
    logger.info(f"Fetching result for {result_id}: {result}")
    
    # Restructure the response if completed to match client expectations
    if result.get("status") == "completed":
        metrics = result.get("metrics", {})
        penalties = metrics.get("penalties", {})
        # Create a new dictionary with metrics at the top level
        response_data = {
            "status": "completed",
            "tasks": result.get("tasks", []),
            "makespan": metrics.get("makespan"),
            "throughput": metrics.get("throughput"),
            "penalties": penalties, # Keep penalties nested or flatten if needed
            # Include other top-level metrics if necessary
        }
        logger.info(f"Returning restructured result for {result_id}: {response_data}")
        return jsonify(response_data)
    else:
        # Return the original result for pending, failed, or not_found statuses
        logger.info(f"Returning original result for {result_id}: {result}")
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)