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
ITERATIONS_PER_TEMP = 10  # Further reduced
MIN_TEMPERATURE = 1
MAX_ITERATIONS = 500  # Further reduced
OPTIMIZATION_TIMEOUT = 5  # Reduced to 5 seconds

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
    def __init__(self, id, name, description, deadline, duration, priority, dependencies, resources, start_time=None):
        self.id = id
        self.name = name
        self.description = description
        self.deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        self.duration = duration
        self.priority = priority
        self.dependencies = dependencies if dependencies else []
        self.resources = resources if resources else []
        self.start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else None

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
        self.is_available = bool(is_available)
        self.cost = cost

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'isAvailable': 1 if self.is_available else 0,
            'cost': self.cost
        }

# CRUD Operations for Tasks
@app.route('/tasks', methods=['GET'])
def get_tasks():
    logger.info("Received GET request for /tasks")
    with get_db() as conn:
        tasks = conn.execute('SELECT * FROM tasks').fetchall()
        return jsonify([dict(task) for task in tasks])

@app.route('/tasks', methods=['POST'])
def create_task():
    logger.info("Received POST request for /tasks")
    data = request.get_json()
    task = Task(
        id=data['id'],
        name=data['name'],
        description=data.get('description', ''),
        deadline=data['deadline'],
        duration=data['duration'],
        priority=data['priority'],
        dependencies=data.get('dependencies', []),
        resources=data.get('resources', []),
        start_time=data.get('startTime')
    )
    with get_db() as conn:
        conn.execute('''
            INSERT INTO tasks (id, name, description, deadline, duration, priority, dependencies, resources, startTime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.id,
            task.name,
            task.description,
            task.deadline.isoformat(),
            task.duration,
            task.priority,
            json.dumps(task.dependencies),
            json.dumps(task.resources),
            task.start_time.isoformat() if task.start_time else None
        ))
        conn.commit()
    return jsonify(task.to_dict()), 201

@app.route('/tasks/<id>', methods=['DELETE'])
def delete_task(id):
    logger.info(f"Received DELETE request for /tasks/{id}")
    with get_db() as conn:
        conn.execute('DELETE FROM tasks WHERE id = ?', (id,))
        conn.commit()
    return jsonify({'message': 'Task deleted'}), 200

# CRUD Operations for Resources
@app.route('/resources', methods=['GET'])
def get_resources():
    logger.info("Received GET request for /resources")
    with get_db() as conn:
        resources = conn.execute('SELECT * FROM resources').fetchall()
        return jsonify([dict(resource) for resource in resources])

@app.route('/resources', methods=['POST'])
def create_resource():
    logger.info("Received POST request for /resources")
    data = request.get_json()
    resource = Resource(
        id=data['id'],
        name=data['name'],
        type=data['type'],
        is_available=data['isAvailable'],
        cost=data['cost']
    )
    with get_db() as conn:
        conn.execute('''
            INSERT INTO resources (id, name, type, isAvailable, cost)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            resource.id,
            resource.name,
            resource.type,
            1 if resource.is_available else 0,
            resource.cost
        ))
        conn.commit()
    return jsonify(resource.to_dict()), 201

@app.route('/resources/<id>', methods=['DELETE'])
def delete_resource(id):
    logger.info(f"Received DELETE request for /resources/{id}")
    with get_db() as conn:
        conn.execute('DELETE FROM resources WHERE id = ?', (id,))
        conn.commit()
    return jsonify({'message': 'Resource deleted'}), 200

# Optimization endpoint
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        logger.info("Received POST request for /optimize")
        data = request.get_json()
        logger.info(f"Optimization request data: {data}")
        tasks_data = data['tasks']
        resources_data = data['resources']
        method = data.get('method', 'PuLP')
        
        tasks = []
        for t in tasks_data:
            try:
                task = Task(
                    id=t['id'],
                    name=t['name'],
                    description=t.get('description', ''),
                    deadline=t['deadline'],
                    duration=t['duration'],
                    priority=t['priority'],
                    dependencies=t.get('dependencies', []),
                    resources=t.get('resources', []),
                    start_time=t.get('startTime')
                )
                if task.duration <= 0:
                    raise ValueError(f"Invalid duration for task {task.id}: {task.duration}")
                if task.deadline < datetime.now():
                    raise ValueError(f"Deadline in the past for task {task.id}: {task.deadline}")
                for dep_id in task.dependencies:
                    if not any(t['id'] == dep_id for t in tasks_data):
                        raise ValueError(f"Invalid dependency {dep_id} for task {task.id}")
                for res_id in task.resources:
                    if not any(r['id'] == res_id for r in resources_data):
                        raise ValueError(f"Invalid resource {res_id} for task {task.id}")
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error processing task {t.get('id', 'unknown')}: {str(e)}")
                return jsonify({'error': f"Invalid task data: {str(e)}"}), 400

        resources = []
        for r in resources_data:
            try:
                resource = Resource(
                    id=r['id'],
                    name=r['name'],
                    type=r['type'],
                    is_available=r['isAvailable'],
                    cost=r['cost']
                )
                if not resource.is_available:
                    logger.warning(f"Resource {resource.id} is not available")
                resources.append(resource)
            except Exception as e:
                logger.error(f"Error processing resource {r.get('id', 'unknown')}: {str(e)}")
                return jsonify({'error': f"Invalid resource data: {str(e)}"}), 400

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
        if method == 'PuLP':
            result = optimize_with_pulp(tasks, resources)
        else:
            result = optimize_with_simulated_annealing(tasks, resources)
        if time.time() - start_time > OPTIMIZATION_TIMEOUT:
            raise TimeoutError("Optimization took too long")
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
        logger.error(f"Optimization failed for result_id {result_id}: {str(e)}")
        with results_lock:
            optimization_results[result_id] = {
                'status': 'error',
                'error': str(e)
            }

# PuLP Optimization
def optimize_with_pulp(tasks, resources):
    try:
        logger.info("Starting PuLP optimization")
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

        logger.info(f"PuLP optimization completed: makespan={makespan_value}")
        return {
            'tasks': tasks,
            'makespan': makespan_value,
            'throughput': throughput,
            'penalties': penalties_dict
        }
    except Exception as e:
        logger.error(f"Error in optimize_with_pulp: {str(e)}")
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
                    logger.warning(f"Invalid resource {res_id} for task {task.id}")

        cost = total_penalty + makespan + resource_cost
        return cost, penalties, makespan

    def generate_initial_solution(tasks, resources):
        now = datetime.now()
        schedule = []
        task_map = {task.id: task for task in tasks}
        visited = set()
        order = []

        def topological_sort(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map[task_id]
            for dep_id in task.dependencies:
                topological_sort(dep_id)
            order.append(task_id)

        for task in tasks:
            topological_sort(task.id)

        earliest_times = {task.id: 0.0 for task in tasks}
        for task_id in order:
            task = task_map[task_id]
            earliest_start = 0.0
            for dep_id in task.dependencies:
                dep_task = task_map[dep_id]
                dep_end = earliest_times[dep_id] + dep_task.duration
                earliest_start = max(earliest_start, dep_end)
            earliest_times[task_id] = earliest_start
            schedule.append(earliest_start)

        logger.info(f"Initial schedule: {schedule}")
        return schedule

    def generate_neighbor(schedule, tasks, resources):
        new_schedule = schedule.copy()
        now = datetime.now()
        T = len(tasks)

        i = random.randint(0, T - 1)
        task = tasks[i]
        earliest_start = 0.0

        for dep_id in task.dependencies:
            for j, dep_task in enumerate(tasks):
                if dep_task.id == dep_id:
                    dep_end_hours = new_schedule[j] + dep_task.duration
                    earliest_start = max(earliest_start, dep_end_hours)
                    break

        deadline_hours = (task.deadline - now).total_seconds() / 3600
        max_start = min(deadline_hours - task.duration, earliest_start + 24)
        if max_start < earliest_start:
            max_start = earliest_start

        new_start = random.uniform(earliest_start, max_start + 0.1)
        new_schedule[i] = new_start

        return new_schedule

    def validate_final_schedule(schedule, tasks):
        if not check_dependencies(schedule, tasks):
            logger.error("Final schedule violates dependencies")
            raise ValueError("Final schedule violates dependencies")
        now = datetime.now()
        for i, task in enumerate(tasks):
            start_time = schedule[i]
            end_time = start_time + task.duration
            deadline_hours = (task.deadline - now).total_seconds() / 3600
            if end_time > deadline_hours:
                logger.warning(f"Task {task.id} misses deadline: end={end_time}, deadline={deadline_hours}")

    # Validate inputs
    if not tasks:
        logger.error("No tasks provided for optimization")
        raise ValueError("No tasks provided for optimization")
    if not resources:
        logger.error("No resources provided for optimization")
        raise ValueError("No resources provided for optimization")

    for task in tasks:
        if task.duration <= 0:
            logger.error(f"Invalid duration for task {task.id}: {task.duration}")
            raise ValueError(f"Invalid duration for task {task.id}: {task.duration}")
        if task.deadline < datetime.now():
            logger.error(f"Deadline in the past for task {task.id}: {task.deadline}")
            raise ValueError(f"Deadline in the past for task {task.id}: {task.deadline}")
        for dep_id in task.dependencies:
            if not any(t.id == dep_id for t in tasks):
                logger.error(f"Invalid dependency {dep_id} for task {task.id}")
                raise ValueError(f"Invalid dependency {dep_id} for task {task.id}")
        for res_id in task.resources:
            if not any(r.id == res_id for r in resources):
                logger.error(f"Invalid resource {res_id} for task {task.id}")
                raise ValueError(f"Invalid resource {res_id} for task {task.id}")

    try:
        now = datetime.now()
        current_schedule = generate_initial_solution(tasks, resources)
        current_cost, current_penalties, current_makespan = calculate_cost(current_schedule, tasks, resources)
        best_schedule = current_schedule
        best_cost = current_cost
        best_penalties = current_penalties
        best_makespan = current_makespan

        temperature = INITIAL_TEMPERATURE
        total_iterations = 0
        while temperature > MIN_TEMPERATURE and total_iterations < MAX_ITERATIONS:
            for _ in range(ITERATIONS_PER_TEMP):
                total_iterations += 1
                neighbor = generate_neighbor(current_schedule, tasks, resources)
                neighbor_cost, neighbor_penalties, neighbor_makespan = calculate_cost(neighbor, tasks, resources)

                cost_diff = neighbor_cost - current_cost
                if cost_diff <= 0 or random.random() < math.exp(-cost_diff / temperature):
                    current_schedule = neighbor
                    current_cost = neighbor_cost
                    current_penalties = neighbor_penalties
                    current_makespan = neighbor_makespan

                    if neighbor_cost < best_cost:
                        best_schedule = neighbor
                        best_cost = neighbor_cost
                        best_penalties = neighbor_penalties
                        best_makespan = neighbor_makespan
                        logger.info(f"New best solution found at iteration {total_iterations}: cost={best_cost}, makespan={best_makespan}")

                if total_iterations >= MAX_ITERATIONS:
                    break
            temperature *= COOLING_RATE

        validate_final_schedule(best_schedule, tasks)

        for i, task in enumerate(tasks):
            start_hours = best_schedule[i]
            task.start_time = now + timedelta(hours=start_hours)

        throughput = len(tasks) / best_makespan if best_makespan > 0 else 0
        logger.info(f"Simulated Annealing completed: makespan={best_makespan}, throughput={throughput}, iterations={total_iterations}")

        return {
            'tasks': tasks,
            'makespan': best_makespan,
            'throughput': throughput,
            'penalties': best_penalties
        }
    except Exception as e:
        logger.error(f"Error in optimize_with_simulated_annealing: {str(e)}")
        raise

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=3000, debug=True)