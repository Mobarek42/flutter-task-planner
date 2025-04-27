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
            print("Column 'start_time' added to table 'tasks'")
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
    print("Task received:", task)
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

def get_initial_solution(tasks, resources):
    now = datetime.datetime.now()
    start_times = {}
    
    # Sort tasks by priority and deadline
    sorted_tasks = sorted(tasks, key=lambda t: (
        -t.get('priority', 0),
        datetime.datetime.fromisoformat(t['deadline'].replace('Z', '+00:00'))
    ))
    
    for task in sorted_tasks:
        if 'startTime' in task and task['startTime']:
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

def evaluate_solution(start_times, tasks, resources):
    """Evaluate the quality of a solution (makespan)."""
    makespan = 0
    for task in tasks:
        task_id = task['id']
        if task_id in start_times:
            start = start_times[task_id]
            end = start + task['duration']
            makespan = max(makespan, end)
    return makespan

def repair_solution(solution, tasks):
    """Repair a solution to respect dependency constraints."""
    for task in tasks:
        task_id = task['id']
        if task_id in solution:
            for dep_id in task.get('dependencies', []):
                if dep_id and dep_id in solution:
                    dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                    if dep_task:
                        min_start = solution[dep_id] + dep_task['duration']
                        solution[task_id] = max(solution[task_id], min_start)

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
    
    # Initialize solution
    current_solution = initial_solution or get_initial_solution(tasks, resources)
    current_cost = evaluate_solution(current_solution, tasks, resources)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    # Simulated annealing algorithm
    temp = initial_temp
    iteration = 0
    no_improvement = 0
    
    while temp > 1 and iteration < max_iterations and no_improvement < no_improvement_limit:
        new_solution = current_solution.copy()
        
        # Select a task to modify (only non-fixed tasks)
        task_ids = [t['id'] for t in tasks if 'startTime' not in t or not t['startTime']]
        if not task_ids:
            break
        task_id = random.choice(task_ids)
        
        # Apply perturbation
        perturbation = random.uniform(-2 * (temp / initial_temp), 2 * (temp / initial_temp))
        new_solution[task_id] += perturbation
        
        # Enforce dependency constraints strictly
        for task in tasks:
            if task['id'] in new_solution:
                for dep_id in task.get('dependencies', []):
                    if dep_id and dep_id in new_solution:
                        dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                        if dep_task:
                            # Ensure task starts after all dependencies are complete
                            min_start = new_solution[dep_id] + dep_task['duration']
                            new_solution[task['id']] = max(new_solution[task['id']], min_start)
        
        # Repair solution to handle resource conflicts
        repair_solution(new_solution, tasks)
        
        # Evaluate new solution
        new_cost = evaluate_solution(new_solution, tasks, resources)
        
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
    
    return best_solution, best_cost, iteration, temp

results = {}

@app.route('/optimize', methods=['POST'])
def optimize_schedule_async():
    data = request.get_json()
    tasks = data['tasks']
    resources = data['resources']
    method = data.get('method', 'PuLP')  # Default to PuLP
    params = data.get('params', {})  # Additional parameters
    result_id = str(uuid.uuid4())
    results[result_id] = {"status": "pending"}

    def optimize_schedule(tasks, resources, method, params, result_id):
        try:
            now = datetime.datetime.now()
            fixed_tasks = [t for t in tasks if 'startTime' in t and t['startTime']]
            tasks_to_optimize = [t for t in tasks if 'startTime' not in t or not t['startTime']]

            if method == 'PuLP':
                prob = LpProblem("Task_Scheduling", LpMinimize)
                start_times = {task['id']: LpVariable(f"start_{task['id']}", 0) for task in tasks_to_optimize}
                makespan = LpVariable("makespan", 0)

                # Makespan constraint
                for task in tasks_to_optimize:
                    prob += makespan >= start_times[task['id']] + task['duration']
                
                # Dependency constraints
                for task in tasks_to_optimize:
                    dependencies = task.get('dependencies', [])
                    for dep_id in dependencies:
                        if dep_id:
                            dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                            if dep_task:
                                if dep_task in fixed_tasks:
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
                    print("Optimal solution found with PuLP")
                    optimized_start_times = {task['id']: value(start_times[task['id']]) for task in tasks_to_optimize}
                    
                    makespan_val = value(makespan)
                    throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                    
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
                else:
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
                best_solution, best_cost, iterations, final_temp = simulated_annealing(
                    tasks, resources, initial_solution=initial_solution, params=sa_params
                )
                
                makespan_val = best_cost
                throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                
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
            else:
                results[result_id] = {"status": "failed", "error": "Invalid optimization method"}
        except Exception as e:
            results[result_id] = {"status": "failed", "error": str(e)}
            print(f"Optimization failed for {result_id}: {e}")

    thread = threading.Thread(target=optimize_schedule, args=(tasks, resources, method, params, result_id))
    thread.start()

    # Return 202 Accepted status code for asynchronous operation
    return jsonify({"status": "pending", "id": result_id}), 202

@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id):
    result = results.get(result_id, {"status": "not_found"})
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

