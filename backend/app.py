from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import pulp
import random
import math
import time
import threading
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulated Annealing parameters
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.95
ITERATIONS_PER_TEMP = 10
MIN_TEMPERATURE = 1
MAX_ITERATIONS = 500
OPTIMIZATION_TIMEOUT = 30  # Augmenté de 5 à 30 secondes pour résoudre le problème de timeout

# In-memory storage for optimization results
optimization_results = {}
optimization_tasks = {}
results_lock = threading.Lock()

# Database connection
def get_db():
    conn = sqlite3.connect('tasks.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                deadline TEXT NOT NULL,
                duration INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                dependencies TEXT,
                resources TEXT,
                startTime TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                isAvailable INTEGER NOT NULL,
                cost REAL NOT NULL
            )
        ''')

# Task model
class Task:
    def __init__(self, id, name, description, deadline, duration, priority, dependencies=None, resources=None, start_time=None):
        self.id = id
        self.name = name
        self.description = description
        self.deadline = deadline
        self.duration = duration
        self.priority = priority
        self.dependencies = dependencies or []
        self.resources = resources or []
        self.start_time = start_time

    @classmethod
    def from_dict(cls, data):
        deadline = datetime.fromisoformat(data['deadline'].replace('Z', '+00:00'))
        start_time = None
        if data.get('startTime'):
            try:
                start_time = datetime.fromisoformat(data['startTime'].replace('Z', '+00:00'))
            except ValueError:
                pass
        
        dependencies = data.get('dependencies', [])
        if isinstance(dependencies, str):
            dependencies = dependencies.split(',') if dependencies else []
        
        resources = data.get('resources', [])
        if isinstance(resources, str):
            try:
                resources = json.loads(resources)
            except json.JSONDecodeError:
                resources = []
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            deadline=deadline,
            duration=int(data['duration']),
            priority=int(data['priority']),
            dependencies=dependencies,
            resources=resources,
            start_time=start_time
        )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'deadline': self.deadline.isoformat(),
            'duration': self.duration,
            'priority': self.priority,
            'dependencies': self.dependencies,
            'resources': self.resources,
            'startTime': self.start_time.isoformat() if self.start_time else None
        }

# Resource model
class Resource:
    def __init__(self, id, name, type, is_available, cost):
        self.id = id
        self.name = name
        self.type = type
        self.is_available = is_available
        self.cost = cost

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            is_available=bool(data.get('isAvailable', True)),
            cost=float(data['cost'])
        )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'isAvailable': 1 if self.is_available else 0,
            'cost': self.cost
        }

# Initialize database on startup
@app.before_first_request
def before_first_request():
    init_db()

# Tasks endpoints
@app.route('/tasks', methods=['GET'])
def get_tasks():
    logger.info("Received GET request for /tasks")
    with get_db() as conn:
        tasks = conn.execute('SELECT * FROM tasks').fetchall()
    return jsonify([dict(task) for task in tasks])

@app.route('/tasks', methods=['POST'])
def add_task():
    logger.info("Received POST request for /tasks")
    data = request.json
    with get_db() as conn:
        conn.execute(
            'INSERT INTO tasks (id, name, description, deadline, duration, priority, dependencies, resources, startTime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                data['id'],
                data['name'],
                data.get('description', ''),
                data['deadline'],
                data['duration'],
                data['priority'],
                json.dumps(data.get('dependencies', [])),
                json.dumps(data.get('resources', [])),
                data.get('startTime')
            )
        )
    return jsonify({'message': 'Task added successfully'}), 201

@app.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    logger.info(f"Received DELETE request for /tasks/{task_id}")
    with get_db() as conn:
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    return jsonify({'message': 'Task deleted successfully'})

# Resources endpoints
@app.route('/resources', methods=['GET'])
def get_resources():
    logger.info("Received GET request for /resources")
    with get_db() as conn:
        resources = conn.execute('SELECT * FROM resources').fetchall()
    return jsonify([dict(resource) for resource in resources])

@app.route('/resources', methods=['POST'])
def add_resource():
    logger.info("Received POST request for /resources")
    data = request.json
    with get_db() as conn:
        conn.execute(
            'INSERT INTO resources (id, name, type, isAvailable, cost) VALUES (?, ?, ?, ?, ?)',
            (
                data['id'],
                data['name'],
                data['type'],
                1 if data.get('isAvailable', True) else 0,
                data['cost']
            )
        )
    return jsonify({'message': 'Resource added successfully'}), 201

@app.route('/resources/<resource_id>', methods=['DELETE'])
def delete_resource(resource_id):
    logger.info(f"Received DELETE request for /resources/{resource_id}")
    with get_db() as conn:
        conn.execute('DELETE FROM resources WHERE id = ?', (resource_id,))
    return jsonify({'message': 'Resource deleted successfully'})

# Optimization endpoint
@app.route('/optimize', methods=['POST'])
def optimize():
    logger.info("Received POST request for /optimize")
    try:
        data = request.json
        method = data.get('method', 'PuLP')
        logger.info(f"Optimization method: {method}")

        tasks_data = data.get('tasks', [])
        resources_data = data.get('resources', [])
        
        tasks = [Task.from_dict(t) for t in tasks_data]
        resources = [Resource.from_dict(r) for r in resources_data]
        
        logger.info(f"Received {len(tasks)} tasks and {len(resources)} resources for optimization")

        def has_circular_dependency(task_id, dependencies, visited, path):
            if task_id in path:
                return True
            if task_id in visited:
                return False
            visited.add(task_id)
            path.add(task_id)
            for dep_id in dependencies:
                for t in tasks:
                    if t.id == dep_id:
                        if has_circular_dependency(t.id, t.dependencies, visited, path):
                            return True
                        break
            path.remove(task_id)
            return False

        for task in tasks:
            if has_circular_dependency(task.id, task.dependencies, set(), set()):
                logger.error(f"Circular dependency detected for task {task.id}")
                return jsonify({'error': f"Circular dependency detected for task {task.id}"}), 400

        result_id = str(time.time())
        with results_lock:
            optimization_results[result_id] = {'status': 'pending', 'id': result_id}
        optimization_tasks[result_id] = threading.Thread(
            target=run_optimization,
            args=(tasks, resources, method, result_id)
        )
        optimization_tasks[result_id].start()
        logger.info(f"Started optimization task {result_id} with method {method}")
        return jsonify({'status': 'pending', 'id': result_id}), 202
    except Exception as e:
        logger.error(f"Error in optimize endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Result retrieval endpoint
@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id):
    logger.info(f"Received GET request for /result/{result_id}")
    with results_lock:
        result = optimization_results.get(result_id, {'status': 'not_found'})
    logger.info(f"Result response for {result_id}: status={result.get('status')}")
    return jsonify(result)

# Run optimization in a separate thread
def run_optimization(tasks, resources, method, result_id):
    try:
        logger.info(f"Running optimization for result_id {result_id} with method {method}")
        start_time = time.time()
        
        # Utiliser un thread avec timeout pour l'optimisation
        optimization_thread = threading.Thread(
            target=run_optimization_with_timeout,
            args=(tasks, resources, method, result_id)
        )
        optimization_thread.daemon = True
        optimization_thread.start()
        optimization_thread.join(OPTIMIZATION_TIMEOUT)
        
        if optimization_thread.is_alive():
            logger.error(f"Optimization timeout for result_id {result_id}")
            with results_lock:
                optimization_results[result_id] = {
                    'status': 'error',
                    'error': "Optimization timeout"
                }
        
    except Exception as e:
        logger.error(f"Optimization failed for result_id {result_id}: {str(e)}")
        with results_lock:
            optimization_results[result_id] = {
                'status': 'error',
                'error': str(e)
            }

# Run optimization with timeout
def run_optimization_with_timeout(tasks, resources, method, result_id):
    try:
        if method == 'PuLP':
            result = optimize_with_pulp(tasks, resources, result_id)  # Ajout du paramètre result_id
        else:
            result = optimize_with_simulated_annealing(tasks, resources)
            
        with results_lock:
            optimization_results[result_id] = {
                'status': 'completed',
                'tasks': [t.to_dict() for t in result['tasks']],
                'makespan': result['makespan'],
                'throughput': result['throughput'],
                'penalties': result['penalties']
            }
        logger.info(f"Optimization completed for result_id {result_id}: makespan={result['makespan']}")
    except Exception as e:
        logger.error(f"Optimization thread failed for result_id {result_id}: {str(e)}")
        with results_lock:
            optimization_results[result_id] = {
                'status': 'error',
                'error': str(e)
            }

# PuLP Optimization
def optimize_with_pulp(tasks, resources, result_id):  # Ajout du paramètre result_id
    try:
        logger.info(f"Starting PuLP optimization for result_id {result_id}")
        prob = pulp.LpProblem("TaskScheduling", pulp.LpMinimize)
        now = datetime.now()
        T = len(tasks)
        R = len(resources)
        M = 10000

        start_times = pulp.LpVariable.dicts("Start", range(T), lowBound=0, cat='Continuous')
        resource_assignments = pulp.LpVariable.dicts("Assign", [(i, j) for i in range(T) for j in range(R)], cat='Binary')
        makespan = pulp.LpVariable("Makespan", lowBound=0, cat='Continuous')

        penalties = pulp.LpVariable.dicts("Penalty", range(T), lowBound=0, cat='Continuous')
        total_penalty = pulp.lpSum([penalties[i] for i in range(T)])
        prob += total_penalty + makespan

        for i in range(T):
            task = tasks[i]
            deadline_hours = (task.deadline - now).total_seconds() / 3600
            prob += start_times[i] >= 0
            prob += start_times[i] + task.duration <= deadline_hours + M * penalties[i]
            prob += start_times[i] + task.duration <= makespan
            prob += penalties[i] >= 0

            for dep_id in task.dependencies:
                for j in range(T):
                    if tasks[j].id == dep_id:
                        prob += start_times[i] >= start_times[j] + tasks[j].duration
                        break

        for i in range(T):
            task = tasks[i]
            if task.resources:
                assigned = pulp.lpSum([resource_assignments[(i, j)] for j in range(R) if resources[j].id in task.resources])
                prob += assigned >= 1
            else:
                logger.warning(f"Task {task.id} has no resources assigned")

        for j in range(R):
            for t in range(T):
                for s in range(t + 1, T):
                    task_t = tasks[t]
                    task_s = tasks[s]
                    if resources[j].id in task_t.resources and resources[j].id in task_s.resources:
                        overlap = pulp.LpVariable(f"Overlap_{t}_{s}_{j}", cat='Binary')
                        prob += start_times[t] + task_t.duration <= start_times[s] + M * (1 - overlap)
                        prob += start_times[s] + task_s.duration <= start_times[t] + M * overlap

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            logger.error(f"PuLP solver failed with status: {pulp.LpStatus[status]}")
            raise ValueError(f"PuLP solver failed with status: {pulp.LpStatus[status]}")

        for i in range(T):
            start_hour = start_times[i].varValue
            if start_hour is not None:
                tasks[i].start_time = now + timedelta(hours=start_hour)
            else:
                logger.error(f"No start time assigned for task {tasks[i].id}")
                raise ValueError(f"No start time assigned for task {tasks[i].id}")

        makespan_value = makespan.varValue if makespan.varValue is not None else 0
        throughput = T / makespan_value if makespan_value > 0 else 0
        penalties_dict = {tasks[i].id: penalties[i].varValue for i in range(T) if penalties[i].varValue is not None}

        logger.info(f"PuLP optimization completed for result_id {result_id}: makespan={makespan_value}")
        return {
            'tasks': tasks,
            'makespan': makespan_value,
            'throughput': throughput,
            'penalties': penalties_dict
        }
    except Exception as e:
        logger.error(f"Error in optimize_with_pulp for result_id {result_id}: {e}")
        raise

# Simulated Annealing Optimization
def optimize_with_simulated_annealing(tasks, resources):
    logger.info(f"Starting Simulated Annealing with {len(tasks)} tasks and {len(resources)} resources")
    
    def calculate_makespan(schedule, tasks):
        if not schedule:
            logger.warning("Empty schedule, returning infinity")
            return float('inf')
        return max((schedule[i] + tasks[i].duration for i in range(len(tasks))), default=0)

    def check_dependencies(schedule, tasks):
        for i, task in enumerate(tasks):
            start_i = schedule[i]
            for dep_id in task.dependencies:
                for j, dep_task in enumerate(tasks):
                    if dep_task.id == dep_id:
                        dep_end = schedule[j] + dep_task.duration
                        if start_i < dep_end:
                            logger.debug(f"Dependency violation: Task {task.id} starts at {start_i}, but depends on {dep_task.id} ending at {dep_end}")
                            return False
                        break
        return True

    def calculate_cost(schedule, tasks, resources):
        makespan = calculate_makespan(schedule, tasks)
        penalties = {}
        total_penalty = 0
        now = datetime.now()

        for i, task in enumerate(tasks):
            start_time = schedule[i]
            end_time = start_time + task.duration
            deadline_hours = (task.deadline - now).total_seconds() / 3600
            penalty = max(0, end_time - deadline_hours) * task.priority * 100
            penalties[task.id] = penalty
            total_penalty += penalty

        if not check_dependencies(schedule, tasks):
            total_penalty += 1000000

        resource_cost = 0
        for i, task in enumerate(tasks):
            for res_id in task.resources:
                res_found = False
                for res in resources:
                    if res.id == res_id:
                        if not res.is_available:
                            total_penalty += 100000
                        resource_cost += res.cost * task.duration
                        res_found = True
                        break
                if not res_found:
                    total_penalty += 100000

        for j, res in enumerate(resources):
            for t in range(len(tasks)):
                for s in range(t + 1, len(tasks)):
                    if res.id in tasks[t].resources and res.id in tasks[s].resources:
                        start_t = schedule[t]
                        end_t = start_t + tasks[t].duration
                        start_s = schedule[s]
                        end_s = start_s + tasks[s].duration
                        if not (end_t <= start_s or end_s <= start_t):
                            total_penalty += 100000

        return makespan + total_penalty / 1000, makespan, penalties

    def generate_initial_solution(tasks):
        schedule = [0] * len(tasks)
        for i, task in enumerate(tasks):
            for dep_id in task.dependencies:
                for j, dep_task in enumerate(tasks):
                    if dep_task.id == dep_id:
                        schedule[i] = max(schedule[i], schedule[j] + dep_task.duration)
                        break
        return schedule

    def generate_neighbor(schedule, tasks, temperature):
        new_schedule = schedule.copy()
        i = random.randint(0, len(tasks) - 1)
        
        # Adjust the perturbation based on temperature
        max_shift = max(1, int(temperature / 100))
        shift = random.randint(-max_shift, max_shift)
        
        new_schedule[i] = max(0, new_schedule[i] + shift)
        
        # Ensure dependencies are respected
        for j, task in enumerate(tasks):
            for dep_id in task.dependencies:
                for k, dep_task in enumerate(tasks):
                    if dep_task.id == dep_id:
                        if new_schedule[j] < new_schedule[k] + dep_task.duration:
                            new_schedule[j] = new_schedule[k] + dep_task.duration
                        break
        
        return new_schedule

    # Initialize
    current_schedule = generate_initial_solution(tasks)
    current_cost, current_makespan, current_penalties = calculate_cost(current_schedule, tasks, resources)
    best_schedule = current_schedule.copy()
    best_cost = current_cost
    best_makespan = current_makespan
    best_penalties = current_penalties.copy()
    
    temperature = INITIAL_TEMPERATURE
    iteration = 0
    
    # Main loop
    while temperature > MIN_TEMPERATURE and iteration < MAX_ITERATIONS:
        for _ in range(ITERATIONS_PER_TEMP):
            new_schedule = generate_neighbor(current_schedule, tasks, temperature)
            new_cost, new_makespan, new_penalties = calculate_cost(new_schedule, tasks, resources)
            
            # Accept or reject
            cost_diff = new_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_schedule = new_schedule
                current_cost = new_cost
                current_makespan = new_makespan
                current_penalties = new_penalties
                
                if current_cost < best_cost:
                    best_schedule = current_schedule.copy()
                    best_cost = current_cost
                    best_makespan = current_makespan
                    best_penalties = current_penalties.copy()
                    logger.info(f"New best solution found: makespan={best_makespan}, cost={best_cost}")
        
        temperature *= COOLING_RATE
        iteration += 1
        logger.debug(f"Iteration {iteration}, Temperature: {temperature}, Best cost: {best_cost}")
    
    # Apply the best schedule to tasks
    now = datetime.now()
    for i, task in enumerate(tasks):
        task.start_time = now + timedelta(hours=best_schedule[i])
    
    throughput = len(tasks) / best_makespan if best_makespan > 0 else 0
    
    logger.info(f"Simulated Annealing completed: makespan={best_makespan}, iterations={iteration}")
    return {
        'tasks': tasks,
        'makespan': best_makespan,
        'throughput': throughput,
        'penalties': best_penalties
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
