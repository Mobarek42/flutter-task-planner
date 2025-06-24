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
import firebase_admin
import os
from firebase_admin import credentials, auth
from functools import wraps
from flask import abort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin
DATABASE = 'tasks.db'

# Initialisation Firebase Admin SDK
cred = credentials.Certificate(os.getenv('FIREBASE_SERVICE_ACCOUNT', 'taskplanner-ap-firebase-adminsdk-fbsvc-e65b4f4134.json'))
firebase_admin.initialize_app(cred)

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

# Décorateur pour vérifier l'authentification
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            abort(401, description='No token provided or invalid token format')
        
        try:
            decoded_token = auth.verify_id_token(token.replace('Bearer ', ''))
            request.user = decoded_token  # Stocker les infos de l'utilisateur pour un usage ultérieur si nécessaire
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            abort(401, description='Invalid or expired token')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/tasks', methods=['GET'])
@require_auth
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
@require_auth
def add_task():
    task = request.get_json()
    logger.info(f"Task received: {task}")
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('INSERT INTO tasks (id, name, description, deadline, duration, priority, dependencies, resources, start_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (task['id'], task['name'], task['description'], task['deadline'], task['duration'],
                      task['priority'], ','.join(task['dependencies']), json.dumps(task['resources']), task.get('startTime')))
    return '', 201

@app.route('/tasks/<task_id>', methods=['DELETE'])
@require_auth
def delete_task(task_id):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    return '', 200

@app.route('/resources', methods=['GET'])
@require_auth
def get_resources():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        resources = conn.execute('SELECT * FROM resources').fetchall()
    return jsonify([dict(resource) for resource in resources])

@app.route('/resources', methods=['POST'])
@require_auth
def add_resource():
    resource = request.get_json()
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('INSERT INTO resources VALUES (?, ?, ?, ?, ?)',
                     (resource['id'], resource['name'], resource['type'], 1 if resource['isAvailable'] else 0, resource['cost']))
    return '', 201

@app.route('/resources/<resource_id>', methods=['DELETE'])
@require_auth
def delete_resource(resource_id):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('DELETE FROM resources WHERE id = ?', (resource_id,))
    return '', 200

def is_task_user_fixed(task):
    """Détermine si une tâche a un startTime fixe défini EXPLICITEMENT par l'utilisateur"""
    return ('startTime' in task and task['startTime'] and 
            not task.get('_computed_start_time', False))

def is_task_fixed(task):
    """Détermine si une tâche a un startTime fixe (défini par l'utilisateur)"""
    return is_task_user_fixed(task)

def get_initial_solution(tasks, resources):
    now = datetime.datetime.now()
    start_times = {}
    
    sorted_tasks = sorted(tasks, key=lambda t: (
        -t.get('priority', 0),
        datetime.datetime.fromisoformat(t['deadline'].replace('Z', '+00:00'))
    ))
    
    for task in sorted_tasks:
        if is_task_fixed(task):
            start_times[task['id']] = (datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00')) - now).total_seconds() / 3600
        else:
            earliest_start = 0
            
            dependencies = task.get('dependencies', [])
            for dep_id in dependencies:
                if dep_id and dep_id in start_times:
                    dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                    if dep_task:
                        dep_end = start_times[dep_id] + dep_task['duration']
                        earliest_start = max(earliest_start, dep_end)
            
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
    if now is None:
        now = datetime.datetime.now()
    
    earliest_start = float('inf')
    latest_end = float('-inf')
    
    for task in tasks:
        task_id = task['id']
        
        if is_task_fixed(task):
            start_time = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
            start_hours = (start_time - now).total_seconds() / 3600
            end_hours = start_hours + task['duration']
            
            earliest_start = min(earliest_start, start_hours)
            latest_end = max(latest_end, end_hours)
        elif task_id in start_times:
            start_hours = start_times[task_id]
            end_hours = start_hours + task['duration']
            
            earliest_start = min(earliest_start, start_hours)
            latest_end = max(latest_end, end_hours)
    
    if earliest_start == float('inf') or latest_end == float('-inf'):
        return 0
    
    makespan = latest_end - earliest_start
    return max(makespan, 0)

def strictly_enforce_dependencies(solution, tasks):
    now = datetime.datetime.now()
    
    task_dict = {task['id']: task for task in tasks}
    
    dep_graph = {}
    for task in tasks:
        task_id = task['id']
        dep_graph[task_id] = task.get('dependencies', [])
    
    visited = set()
    temp_visited = set()
    order = []
    
    def visit(task_id):
        if task_id in temp_visited:
            return
        if task_id in visited:
            return
        
        temp_visited.add(task_id)
        for dep_id in dep_graph.get(task_id, []):
            if dep_id:
                visit(dep_id)
        
        temp_visited.remove(task_id)
        visited.add(task_id)
        order.append(task_id)
    
    for task_id in dep_graph:
        if task_id not in visited:
            visit(task_id)
    
    for task_id in reversed(order):
        if task_id in solution:
            start_time = solution[task_id]
            
            dependencies = dep_graph.get(task_id, [])
            for dep_id in dependencies:
                if dep_id:
                    dep_task = task_dict[dep_id]
                    
                    if is_task_fixed(dep_task):
                        dep_start_abs = datetime.datetime.fromisoformat(dep_task['startTime'].replace('Z', '+00:00'))
                        dep_end_abs = dep_start_abs + datetime.timedelta(hours=dep_task['duration'])
                        dep_end_relative = (dep_end_abs - now).total_seconds() / 3600
                        
                        if start_time < dep_end_relative:
                            solution[task_id] = dep_end_relative
                    elif dep_id in solution:
                        dep_duration = dep_task['duration']
                        dep_end_time = solution[dep_id] + dep_duration
                        
                        if start_time < dep_end_time:
                            solution[task_id] = dep_end_time

def repair_solution(solution, tasks):
    strictly_enforce_dependencies(solution, tasks)
    
    task_dict = {task['id']: task for task in tasks}
    
    has_conflicts = True
    max_iterations = 100
    iteration = 0
    
    while has_conflicts and iteration < max_iterations:
        has_conflicts = False
        iteration += 1
        
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
                
                if not (task1_end <= task2_start or task2_end <= task1_start):
                    shared_resources = set(task1_resources).intersection(set(task2_resources))
                    if shared_resources:
                        has_conflicts = True
                        
                        task1_priority = task1.get('priority', 0)
                        task2_priority = task2.get('priority', 0)
                        
                        if task1_priority > task2_priority:
                            solution[task2_id] = task1_end
                        elif task2_priority > task1_priority:
                            solution[task1_id] = task2_end
                        else:
                            task1_deadline = datetime.datetime.fromisoformat(task1['deadline'].replace('Z', '+00:00'))
                            task2_deadline = datetime.datetime.fromisoformat(task2['deadline'].replace('Z', '+00:00'))
                            
                            if task1_deadline > task2_deadline:
                                solution[task1_id] = task2_end
                            else:
                                solution[task2_id] = task1_end
        
        strictly_enforce_dependencies(solution, tasks)
    
    return solution

def simulated_annealing(tasks, resources, initial_solution=None, params=None):
    params = params or {}
    initial_temp = params.get('initial_temp', 1000.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    max_iterations = params.get('max_iterations', 10000)
    no_improvement_limit = params.get('no_improvement_limit', 1000)
    
    now = datetime.datetime.now()
    
    task_dict = {task['id']: task for task in tasks}
    
    current_solution = initial_solution or get_initial_solution(tasks, resources)
    
    strictly_enforce_dependencies(current_solution, tasks)
    
    current_cost = evaluate_solution(current_solution, tasks, resources, now)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    temp = initial_temp
    iteration = 0
    no_improvement = 0
    
    while temp > 1 and iteration < max_iterations and no_improvement < no_improvement_limit:
        new_solution = current_solution.copy()
        
        task_ids = [t['id'] for t in tasks if not is_task_fixed(t)]
        if not task_ids:
            break
        task_id = random.choice(task_ids)
        
        perturbation = random.uniform(-2 * (temp / initial_temp), 2 * (temp / initial_temp))
        new_solution[task_id] += perturbation
        
        new_solution[task_id] = max(0, new_solution[task_id])
        
        strictly_enforce_dependencies(new_solution, tasks)
        repair_solution(new_solution, tasks)
        
        new_cost = evaluate_solution(new_solution, tasks, resources, now)
        
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
        
        temp *= cooling_rate
        iteration += 1
    
    strictly_enforce_dependencies(best_solution, tasks)
    
    final_cost = evaluate_solution(best_solution, tasks, resources, now)
    
    return best_solution, final_cost, iteration, temp

results = {}

@app.route('/optimize', methods=['POST'])
@require_auth
def optimize_schedule_async():
    data = request.get_json()
    tasks = copy.deepcopy(data['tasks'])
    resources = data['resources']
    method = data.get('method', 'PuLP')
    params = data.get('params', {})

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
            fixed_tasks = [t for t in tasks if is_task_fixed(t)]
            tasks_to_optimize = [t for t in tasks if not is_task_fixed(t)]

            logger.info(f"Fixed tasks: {len(fixed_tasks)}, Tasks to optimize: {len(tasks_to_optimize)} for method {method}")

            if method == 'PuLP':
                if len(tasks_to_optimize) == 0:
                    logger.warning(f"No tasks to optimize for PuLP method (result_id: {result_id})")
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

                for task in tasks_to_optimize:
                    prob += makespan >= start_times[task['id']] + task['duration']

                for task in fixed_tasks:
                    start_time = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
                    end_time = start_time + datetime.timedelta(hours=task['duration'])
                    task_end_hours = (end_time - now).total_seconds() / 3600
                    prob += makespan >= task_end_hours

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

                if params.get('enforce_deadlines', False):
                    for task in tasks_to_optimize:
                        deadline_str = task['deadline'].replace('Z', '')
                        if '+' in deadline_str:
                            deadline_str = deadline_str.split('+')[0]
                        deadline = datetime.datetime.fromisoformat(deadline_str)
                        deadline_hours = (deadline - now).total_seconds() / 3600
                        prob += start_times[task['id']] + task['duration'] <= deadline_hours

                prob += makespan

                prob.solve()
                if LpStatus[prob.status] == 'Optimal':
                    logger.info(f"Optimal solution found with PuLP for result_id {result_id}")
                    optimized_start_times = {task['id']: value(start_times[task['id']]) for task in tasks_to_optimize}

                    makespan_val = value(makespan)
                    throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                    logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id}")

                    metrics = {
                        "makespan": makespan_val,
                        "throughput": throughput,
                    }

                    penalties = {
                        "deadline_violations": 0.0,
                        "dependency_violations": 0.0,
                        "resource_conflicts": 0.0
                    }

                    final_tasks = []
                    for task in tasks:
                        if task['id'] in optimized_start_times:
                            start_hour = optimized_start_times[task['id']]
                            start_time = now + datetime.timedelta(hours=start_hour)
                            task['startTime'] = start_time.isoformat()
                            task['_computed_start_time'] = True
                            end = start_time + datetime.timedelta(hours=task['duration'])
                            deadline_str = task['deadline'].replace('Z', '')
                            if '+' in deadline_str:
                                deadline_str = deadline_str.split('+')[0]
                            deadline = datetime.datetime.fromisoformat(deadline_str)
                            if end > deadline:
                                penalties["deadline_violations"] += (end - deadline).total_seconds() / 3600
                        final_tasks.append(task)

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
                strictly_enforce_dependencies(initial_solution, tasks)

                best_solution, best_cost, iterations, final_temp = simulated_annealing(
                    tasks, resources, initial_solution=initial_solution, params=sa_params
                )

                makespan_val = best_cost
                throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id} using SA")

                metrics = {
                    "makespan": makespan_val,
                    "throughput": throughput,
                    "iterations": iterations,
                    "final_temperature": final_temp
                }

                penalties = {
                    "deadline_violations": 0.0,
                    "dependency_violations": 0.0,
                    "resource_conflicts": 0.0
                }

                final_tasks = []
                for task in tasks:
                    if task['id'] in best_solution:
                        start_hour = best_solution[task['id']]
                        start_time = now + datetime.timedelta(hours=start_hour)
                        task['startTime'] = start_time.isoformat()
                        task['_computed_start_time'] = True
                        end = start_time + datetime.timedelta(hours=task['duration'])
                        deadline_str = task['deadline'].replace('Z', '')
                        if '+' in deadline_str:
                            deadline_str = deadline_str.split('+')[0]
                        deadline = datetime.datetime.fromisoformat(deadline_str)
                        if end > deadline:
                            penalties["deadline_violations"] += (end - deadline).total_seconds() / 3600
                    final_tasks.append(task)

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

    return jsonify({"status": "pending", "id": result_id}), 202

@app.route('/result/<result_id>', methods=['GET'])
@require_auth
def get_result(result_id):
    result = results.get(result_id, {"status": "not_found"})
    logger.info(f"Fetching result for {result_id}: {result}")

    if result.get("status") == "completed":
        metrics = result.get("metrics", {})
        penalties = metrics.get("penalties", {})
        response_data = {
            "status": "completed",
            "tasks": result.get("tasks", []),
            "makespan": metrics.get("makespan"),
            "throughput": metrics.get("throughput"),
            "penalties": penalties,
        }
        logger.info(f"Returning restructured result for {result_id}: {response_data}")
        return jsonify(response_data)
    else:
        logger.info(f"Returning original result for {result_id}: {result}")
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)