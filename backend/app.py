import sqlite3
import datetime
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
# import pulp # Ensure pulp is imported if using pulp.PULP_CBC_CMD
import random
import math
import threading
import uuid
import logging

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

def get_initial_solution(tasks, resources):
    now = datetime.datetime.now()
    start_times = {}
    task_map = {t['id']: t for t in tasks}
    
    # Sort tasks by priority and deadline
    sorted_tasks = sorted(tasks, key=lambda t: (
        -t.get('priority', 0),
        datetime.datetime.fromisoformat(t['deadline'].replace('Z', '+00:00'))
    ))
    
    for task in sorted_tasks:
        task_id = task['id']
        if 'startTime' in task and task['startTime']:
            start_times[task_id] = (datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00')) - now).total_seconds() / 3600
        else:
            earliest_start = 0
            
            # Respect dependencies
            dependencies = task.get('dependencies', [])
            for dep_id in dependencies:
                if dep_id and dep_id in start_times:
                    dep_task = task_map.get(dep_id)
                    if dep_task:
                        dep_end = start_times[dep_id] + dep_task['duration']
                        earliest_start = max(earliest_start, dep_end)
            
            # Simple resource conflict avoidance (greedy)
            task_resources = task.get('resources', [])
            current_time = earliest_start
            conflict = True
            max_attempts = len(tasks) * len(resources) + 1 # Limit attempts to avoid infinite loops
            attempts = 0
            while conflict and attempts < max_attempts:
                attempts += 1
                conflict = False
                max_resource_end_time = current_time # Time when all needed resources are free
                for res_id in task_resources:
                    latest_end_time_for_res = 0
                    for other_task_id, other_start_time in start_times.items():
                        other_task = task_map.get(other_task_id)
                        if other_task and res_id in other_task.get('resources', []):
                            other_end_time = other_start_time + other_task['duration']
                            # Check for overlap
                            if max(current_time, other_start_time) < min(current_time + task['duration'], other_end_time):
                                latest_end_time_for_res = max(latest_end_time_for_res, other_end_time)
                    
                    if latest_end_time_for_res > current_time:
                        max_resource_end_time = max(max_resource_end_time, latest_end_time_for_res)
                        conflict = True
                
                if conflict:
                    current_time = max_resource_end_time # Try starting after the conflict
                else:
                    earliest_start = current_time # Found a conflict-free start time
            
            if attempts >= max_attempts:
                 logger.warning(f"Could not resolve resource conflict for task {task_id} in initial solution after {attempts} attempts.")
                 # Assign earliest possible time despite conflict, SA might resolve it
                 earliest_start = current_time 

            start_times[task_id] = earliest_start
    
    # Final dependency check on initial solution
    repair_solution(start_times, tasks)
    return start_times

def evaluate_solution(start_times, tasks, resources):
    """Evaluate the quality of a solution (makespan)."""
    makespan = 0
    for task in tasks:
        task_id = task['id']
        if task_id in start_times:
            start = start_times[task_id]
            # Ensure duration is treated as a number
            duration = task.get('duration', 0)
            if not isinstance(duration, (int, float)):
                try:
                    duration = float(duration)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid duration for task {task_id}: {duration}. Using 0.")
                    duration = 0
            end = start + duration
            makespan = max(makespan, end)
    return makespan

def repair_solution(solution, tasks):
    """Repair a solution to respect dependency constraints iteratively."""
    task_map = {t['id']: t for t in tasks}
    changed = True
    passes = 0
    max_passes = len(tasks) + 1 # Limit passes to avoid infinite loops

    while changed and passes < max_passes:
        changed = False
        passes += 1
        # Iterate through tasks (order might matter, but simple iteration often works)
        for task in tasks:
            task_id = task['id']
            if task_id in solution:
                min_start_due_to_deps = 0
                dependencies = task.get('dependencies', [])
                for dep_id in dependencies:
                    if dep_id and dep_id in solution:
                        dep_task = task_map.get(dep_id)
                        if dep_task:
                            # Ensure duration is treated as a number
                            dep_duration = dep_task.get('duration', 0)
                            if not isinstance(dep_duration, (int, float)):
                                try:
                                    dep_duration = float(dep_duration)
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid duration for dependency task {dep_id}: {dep_duration}. Using 0.")
                                    dep_duration = 0
                            dep_end_time = solution[dep_id] + dep_duration
                            min_start_due_to_deps = max(min_start_due_to_deps, dep_end_time)
                
                # Ensure start time respects dependencies and is non-negative
                required_start_time = max(0, min_start_due_to_deps)
                if solution[task_id] < required_start_time:
                    solution[task_id] = required_start_time
                    changed = True # Mark that a change was made in this pass
    
    if passes >= max_passes and changed:
        logger.warning("Repair solution reached max passes, potential dependency cycle or complex issue.")

def simulated_annealing(tasks, resources, initial_solution=None, params=None):
    """
    Optimize the schedule using simulated annealing.
    Minimal fix: Ensures dependency constraints are checked iteratively after perturbation.
    """
    params = params or {}
    initial_temp = params.get('initial_temp', 1000.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    max_iterations = params.get('max_iterations', 10000)
    no_improvement_limit = params.get('no_improvement_limit', 1000)
    
    current_solution = initial_solution or get_initial_solution(tasks, resources)
    # Ensure initial solution is valid
    repair_solution(current_solution, tasks)
    current_cost = evaluate_solution(current_solution, tasks, resources)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    temp = initial_temp
    iteration = 0
    no_improvement = 0
    
    task_map = {t['id']: t for t in tasks}
    tasks_to_optimize_ids = [t['id'] for t in tasks if 'startTime' not in t or not t['startTime']]

    while temp > 1 and iteration < max_iterations and no_improvement < no_improvement_limit:
        if not tasks_to_optimize_ids:
            logger.info("No tasks to optimize in SA.")
            break
            
        new_solution = current_solution.copy()
        
        # Select a task to modify
        task_id = random.choice(tasks_to_optimize_ids)
        
        # Apply perturbation (shift start time)
        perturbation = random.uniform(-temp / 100, temp / 100) # Smaller perturbation range
        proposed_start_time = new_solution[task_id] + perturbation
        
        # Ensure the new start time is non-negative
        new_solution[task_id] = max(0, proposed_start_time)
        
        # *** Minimal Fix: Call iterative repair_solution AFTER perturbation ***
        repair_solution(new_solution, tasks)
        # *******************************************************************

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
                # logger.debug(f"SA New best cost: {best_cost} at iter {iteration}") # Optional debug log
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        # Cooling
        temp *= cooling_rate
        iteration += 1
        
        # Optional progress log
        # if iteration % 500 == 0:
        #      logger.debug(f"SA Iter: {iteration}, Temp: {temp:.2f}, Cost: {current_cost:.2f}, Best: {best_cost:.2f}")

    logger.info(f"SA finished. Iter: {iteration}, Temp: {temp:.2f}, Best Cost: {best_cost:.2f}")
    # Final repair on the best solution found
    repair_solution(best_solution, tasks)
    best_cost = evaluate_solution(best_solution, tasks, resources) # Re-evaluate after final repair
    logger.info(f"SA final repaired Best Cost: {best_cost:.2f}")

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
            # Convert deadline/startTime strings to datetime objects for internal use if needed
            for task in tasks:
                if isinstance(task.get('deadline'), str):
                    try:
                        task['deadline_dt'] = datetime.datetime.fromisoformat(task['deadline'].replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Invalid deadline format for task {task['id']}: {task['deadline']}")
                        task['deadline_dt'] = now + datetime.timedelta(days=365) # Default
                if isinstance(task.get('startTime'), str):
                     try:
                        task['startTime_dt'] = datetime.datetime.fromisoformat(task['startTime'].replace('Z', '+00:00'))
                     except ValueError:
                        logger.warning(f"Invalid startTime format for task {task['id']}: {task['startTime']}")
                        # Handle invalid start time - perhaps remove it?
                        task.pop('startTime', None)
                        task.pop('startTime_dt', None)

            if method == 'PuLP':
                # --- PuLP Optimization Logic --- 
                # Assuming pulp is correctly installed and potentially imported as needed
                try:
                    import pulp 
                except ImportError:
                     logger.error("PuLP library not found. Please install it: pip install pulp")
                     results[result_id] = {"status": "failed", "error": "PuLP library not installed"}
                     return
                     
                prob = LpProblem("Task_Scheduling", LpMinimize)
                start_times = {task['id']: LpVariable(f"start_{task['id']}", 0) for task in tasks if 'startTime' not in task or not task['startTime']}
                makespan = LpVariable("makespan", 0)
                task_map = {t['id']: t for t in tasks}
                tasks_to_optimize = [t for t in tasks if 'startTime' not in t or not t['startTime']]

                # Makespan constraint
                for task in tasks_to_optimize:
                    prob += makespan >= start_times[task['id']] + task['duration']
                
                # Dependency constraints
                for task in tasks_to_optimize:
                    dependencies = task.get('dependencies', [])
                    for dep_id in dependencies:
                        if dep_id:
                            dep_task = task_map.get(dep_id)
                            if dep_task:
                                if 'startTime' in dep_task and dep_task['startTime']:
                                    dep_start_dt = dep_task['startTime_dt']
                                    dep_end_hours = (dep_start_dt - now).total_seconds() / 3600 + dep_task['duration']
                                    prob += start_times[task['id']] >= dep_end_hours
                                elif dep_id in start_times:
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
                            if t1['id'] in start_times and t2['id'] in start_times:
                                b = LpVariable(f"b_{t1['id']}_{t2['id']}", cat='Binary')
                                M = sum(t['duration'] for t in tasks) + 1 # A large number
                                prob += start_times[t1['id']] + t1['duration'] <= start_times[t2['id']] + M * (1 - b)
                                prob += start_times[t2['id']] + t2['duration'] <= start_times[t1['id']] + M * b

                # Deadline constraints (optional)
                if params.get('enforce_deadlines', False):
                    for task in tasks_to_optimize:
                        if 'deadline_dt' in task:
                            deadline_hours = (task['deadline_dt'] - now).total_seconds() / 3600
                            prob += start_times[task['id']] + task['duration'] <= deadline_hours

                # Objective function: minimize makespan
                prob += makespan

                # Set a time limit for the solver
                time_limit = params.get('pulp_time_limit', 30) # Default 30 seconds
                solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1)
                prob.solve(solver)

                if LpStatus[prob.status] == 'Optimal' or LpStatus[prob.status] == 'Feasible':
                    status_msg = "Optimal" if LpStatus[prob.status] == 'Optimal' else "Feasible"
                    logger.info(f"{status_msg} solution found with PuLP for result_id {result_id}")
                    optimized_start_times = {task['id']: value(start_times[task['id']]) for task in tasks_to_optimize if task['id'] in start_times and start_times[task['id']].varValue is not None}
                    
                    # Check if makespan has a value
                    makespan_val = value(makespan) if makespan.varValue is not None else evaluate_solution(optimized_start_times, tasks, resources)
                    throughput = len(tasks) / makespan_val if makespan_val > 0 else 0
                    logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id}")
                    
                    metrics = {"makespan": makespan_val, "throughput": throughput}
                    penalties = {"deadline_violations": 0.0, "dependency_violations": 0.0, "resource_conflicts": 0.0}
                    
                    final_tasks = []
                    for task in tasks:
                        task_copy = task.copy()
                        if task_copy['id'] in optimized_start_times:
                            start_hour = optimized_start_times[task_copy['id']]
                            start_time = now + datetime.timedelta(hours=start_hour)
                            task_copy['startTime'] = start_time.isoformat() + 'Z'
                            end_time = start_time + datetime.timedelta(hours=task_copy['duration'])
                            if 'deadline_dt' in task_copy and end_time > task_copy['deadline_dt']:
                                penalties["deadline_violations"] += (end_time - task_copy['deadline_dt']).total_seconds() / 3600
                        task_copy.pop('deadline_dt', None)
                        task_copy.pop('startTime_dt', None)
                        final_tasks.append(task_copy)
                    
                    metrics["resource_utilization"] = {} # Placeholder
                    metrics["penalties"] = penalties
                    
                    results[result_id] = {"status": "completed", "tasks": final_tasks, "metrics": metrics}
                    logger.info(f"Stored result for {result_id}")
                else:
                    logger.error(f"PuLP failed for result_id {result_id} (status: {LpStatus[prob.status]})")
                    results[result_id] = {"status": "failed", "error": f"PuLP failed (status: {LpStatus[prob.status]})"}
                    return
            # --- End PuLP --- 

            elif method == 'SimulatedAnnealing':
                logger.info(f"Starting Simulated Annealing for result_id {result_id}")
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
                logger.info(f"Calculated makespan: {makespan_val}, throughput: {throughput} for result_id {result_id} using SA")
                
                metrics = {
                    "makespan": makespan_val,
                    "throughput": throughput,
                    "iterations": iterations,
                    "final_temperature": final_temp
                }
                penalties = {"deadline_violations": 0.0, "dependency_violations": 0.0, "resource_conflicts": 0.0}
                
                final_tasks = []
                for task in tasks:
                    task_copy = task.copy()
                    if task_copy['id'] in best_solution:
                        start_hour = best_solution[task_copy['id']]
                        start_time = now + datetime.timedelta(hours=start_hour)
                        task_copy['startTime'] = start_time.isoformat() + 'Z'
                        end_time = start_time + datetime.timedelta(hours=task_copy['duration'])
                        if 'deadline_dt' in task_copy and end_time > task_copy['deadline_dt']:
                            penalties["deadline_violations"] += (end_time - task_copy['deadline_dt']).total_seconds() / 3600
                    task_copy.pop('deadline_dt', None)
                    task_copy.pop('startTime_dt', None)
                    final_tasks.append(task_copy)
                
                metrics["resource_utilization"] = {} # Placeholder
                metrics["penalties"] = penalties
                
                results[result_id] = {"status": "completed", "tasks": final_tasks, "metrics": metrics}
                logger.info(f"Stored result for {result_id}")
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
    
    if result.get("status") == "completed":
        metrics = result.get("metrics", {})
        response_data = {
            "status": "completed",
            "tasks": result.get("tasks", []),
            "makespan": metrics.get("makespan"),
            "throughput": metrics.get("throughput"),
            "metrics": metrics 
        }
        return jsonify(response_data)
    else:
        return jsonify(result)

if __name__ == '__main__':
    # Make sure to install necessary packages: pip install Flask Flask-Cors pulp
    app.run(host='0.0.0.0', port=3000, debug=True)

