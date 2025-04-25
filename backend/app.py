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
OPTIMIZATION_TIMEOUT = 5

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
        # Create the table only if it doesn't exist
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
                start_time TEXT
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
        conn.commit()
        logger.info("Database initialized with updated schema")

# Migrate database to add missing columns
def migrate_db():
    with get_db() as conn:
        try:
            # Check and migrate tasks table
            cursor = conn.execute('PRAGMA table_info(tasks)')
            columns = [info[1] for info in cursor.fetchall()]
            
            # If task_name exists, rename it to name
            if 'task_name' in columns:
                logger.info("Renaming column task_name to name in tasks table")
                conn.execute('ALTER TABLE tasks RENAME COLUMN task_name TO name')
                conn.commit()
            
            # Add missing columns if they don't exist
            if 'start_time' not in columns:
                logger.info("Adding start_time column to tasks table")
                conn.execute('ALTER TABLE tasks ADD COLUMN start_time TEXT')
                conn.commit()
                logger.info("Migration completed: start_time column added to tasks")
            else:
                logger.info("start_time column already exists in tasks table")

            # Check and migrate resources table
            cursor = conn.execute('PRAGMA table_info(resources)')
            columns = [info[1] for info in cursor.fetchall()]
            if 'isAvailable' not in columns:
                logger.info("Adding isAvailable column to resources table")
                conn.execute('ALTER TABLE resources ADD COLUMN isAvailable INTEGER NOT NULL DEFAULT 1')
                conn.commit()
                logger.info("Migration completed: isAvailable column added to resources")
            else:
                logger.info("isAvailable column already exists in resources table")
        except sqlite3.Error as e:
            logger.error(f"Error during database migration: {e}")
            # If migration fails, drop the tasks table and recreate it
            logger.info("Dropping and recreating tasks table due to migration failure")
            conn.execute('DROP TABLE IF EXISTS tasks')
            conn.execute('''
                CREATE TABLE tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    deadline TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    dependencies TEXT,
                    resources TEXT,
                    start_time TEXT
                )
            ''')
            conn.commit()
            logger.info("Tasks table recreated successfully")

# Task model
class Task:
    def __init__(self, id, name, description, deadline, duration, priority, dependencies, resources, start_time=None):
        self.id = id
        self.name = name
        self.description = description
        self.deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00')) if isinstance(deadline, str) else deadline
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
        tasks_list = []
        for task in tasks:
            task_dict = dict(task)
            task_dict['dependencies'] = json.loads(task_dict['dependencies'] or '[]')
            task_dict['resources'] = json.loads(task_dict['resources'] or '[]')
            task_dict['startTime'] = task_dict['start_time']
            tasks_list.append(task_dict)
        return jsonify(tasks_list)

@app.route('/tasks', methods=['POST'])
def create_task():
    logger.info("Received POST request for /tasks")
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['id', 'name', 'deadline', 'duration', 'priority']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f"Missing required field: {field}"}), 400

        # Validate data types
        try:
            duration = int(data['duration'])
            priority = int(data['priority'])
            if duration <= 0:
                logger.error(f"Invalid duration: {duration}")
                return jsonify({'error': 'Duration must be a positive integer'}), 400
            if priority <= 0:
                logger.error(f"Invalid priority: {priority}")
                return jsonify({'error': 'Priority must be a positive integer'}), 400
        except (ValueError, TypeError):
            logger.error("Invalid type for duration or priority")
            return jsonify({'error': 'Duration and priority must be integers'}), 400

        # Validate deadline format
        try:
            deadline_dt = datetime.fromisoformat(data['deadline'].replace('Z', '+00:00'))
            if deadline_dt < datetime.now():
                logger.error(f"Deadline in the past: {data['deadline']}")
                return jsonify({'error': 'Deadline cannot be in the past'}), 400
        except ValueError:
            logger.error(f"Invalid deadline format: {data['deadline']}")
            return jsonify({'error': 'Invalid deadline format'}), 400

        # Validate startTime if provided
        start_time = data.get('startTime')
        if start_time:
            try:
                datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                logger.error(f"Invalid startTime format: {start_time}")
                return jsonify({'error': 'Invalid startTime format'}), 400

        task = Task(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            deadline=data['deadline'],
            duration=duration,
            priority=priority,
            dependencies=data.get('dependencies', []),
            resources=data.get('resources', []),
            start_time=start_time
        )

        with get_db() as conn:
            try:
                conn.execute('''
                    INSERT INTO tasks (id, name, description, deadline, duration, priority, dependencies, resources, start_time)
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
            except sqlite3.IntegrityError:
                logger.error(f"Task ID {task.id} already exists")
                return jsonify({'error': f"Task ID {task.id} already exists"}), 400
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
                return jsonify({'error': f"Database error: {e}"}), 500

        logger.info(f"Task created successfully: {task.id}")
        return jsonify(task.to_dict()), 201
    except Exception as e:
        logger.error(f"Error in create_task: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tasks/<id>', methods=['DELETE'])
def delete_task(id):
    logger.info(f"Received DELETE request for /tasks/{id}")
    with get_db() as conn:
        cursor = conn.execute('DELETE FROM tasks WHERE id = ?', (id,))
        conn.commit()
        if cursor.rowcount == 0:
            logger.warning(f"Task {id} not found")
            return jsonify({'error': 'Task not found'}), 404
    logger.info(f"Task deleted: {id}")
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
    try:
        data = request.get_json()
        logger.info(f"Received JSON: {data}")
        if not data:
            logger.error("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['id', 'name', 'type', 'isAvailable', 'cost']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f"Missing required field: {field}"}), 400

        # Flexible validation for isAvailable
        is_available = data['isAvailable']
        if isinstance(is_available, str):
            is_available = is_available.lower() in ('true', '1', 'yes')
        elif not isinstance(is_available, bool):
            logger.error(f"Invalid isAvailable value: {is_available}")
            return jsonify({'error': 'isAvailable must be a boolean or string (true/false)'}), 400

        # Flexible validation for cost
        try:
            cost = float(data['cost'])
            if cost < 0:
                logger.error(f"Invalid cost: {cost}")
                return jsonify({'error': 'Cost must be non-negative'}), 400
        except (ValueError, TypeError):
            logger.error(f"Invalid cost value: {data['cost']}")
            return jsonify({'error': 'Cost must be a number'}), 400

        resource = Resource(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            is_available=is_available,
            cost=cost
        )

        with get_db() as conn:
            try:
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
            except sqlite3.IntegrityError:
                logger.error(f"Resource ID {resource.id} already exists")
                return jsonify({'error': f"Resource ID {resource.id} already exists"}), 400
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
                return jsonify({'error': f"Database error: {e}"}), 500

        logger.info(f"Resource created successfully: {resource.id}")
        return jsonify(resource.to_dict()), 201
    except Exception as e:
        logger.error(f"Error in create_resource: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/resources/<id>', methods=['DELETE'])
def delete_resource(id):
    logger.info(f"Received DELETE request for /resources/{id}")
    with get_db() as conn:
        cursor = conn.execute('SELECT id FROM resources WHERE id = ?', (id,))
        resource = cursor.fetchone()
        if not resource:
            logger.warning(f"Resource {id} not found in database")
            return jsonify({'error': 'Resource not found'}), 404
        cursor = conn.execute('DELETE FROM resources WHERE id = ?', (id,))
        conn.commit()
        if cursor.rowcount == 0:
            logger.warning(f"Failed to delete resource {id} (no rows affected)")
            return jsonify({'error': 'Resource not found'}), 404
    logger.info(f"Resource deleted: {id}")
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

        # Validate inputs to prevent solver crashes
        for task in tasks:
            if task.duration <= 0:
                raise ValueError(f"Task {task.id} has invalid duration: {task.duration}")
            deadline_hours = (task.deadline - now).total_seconds() / 3600
            if deadline_hours < 0:
                raise ValueError(f"Task {task.id} has a deadline in the past: {task.deadline}")
            if deadline_hours < task.duration:
                logger.warning(f"Task {task.id} has infeasible deadline: {deadline_hours}h < {task.duration}h duration")
                # Adjust deadline to make the problem feasible
                task.deadline = now + timedelta(hours=task.duration + 1)

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

        # Solve with a time limit to prevent hanging
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
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
        logger.error(f"Error in optimize_with_pulp: {e}")
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
        logger.error(f"Error in optimize_with_simulated_annealing: {e}")
        raise

if __name__ == '__main__':
    init_db()
    migrate_db()
    app.run(host='0.0.0.0', port=3000, debug=True)