from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import datetime
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Backend de planification des tâches actif !"

def get_db():
    conn = sqlite3.connect('tasks.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                deadline TEXT,
                duration INTEGER,
                priority INTEGER,
                startTime TEXT,
                dependencies TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                isAvailable INTEGER,
                cost REAL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS task_resources (
                taskId TEXT,
                resourceId TEXT,
                FOREIGN KEY(taskId) REFERENCES tasks(id),
                FOREIGN KEY(resourceId) REFERENCES resources(id)
            )
        ''')
        conn.commit()

init_db()

@app.route('/tasks', methods=['GET'])
def get_tasks():
    with get_db() as conn:
        tasks = conn.execute('SELECT * FROM tasks').fetchall()
        task_list = [dict(task) for task in tasks]
        for task in task_list:
            resources = conn.execute('SELECT resourceId FROM task_resources WHERE taskId = ?', (task['id'],)).fetchall()
            task['resources'] = [r['resourceId'] for r in resources]
        return jsonify(task_list)

@app.route('/tasks', methods=['POST'])
def add_task():
    data = request.get_json()
    task = {
        'id': data['id'],
        'name': data['name'],
        'description': data['description'],
        'deadline': data['deadline'],
        'duration': data['duration'],
        'priority': data['priority'],
        'startTime': None,
        'dependencies': ','.join(data['dependencies'])
    }
    resources = data.get('resources', [])
    with get_db() as conn:
        conn.execute('''
            INSERT INTO tasks (id, name, description, deadline, duration, priority, startTime, dependencies)
            VALUES (:id, :name, :description, :deadline, :duration, :priority, :startTime, :dependencies)
        ''', task)
        for res_id in resources:
            conn.execute('INSERT INTO task_resources (taskId, resourceId) VALUES (?, ?)', (task['id'], res_id))
        conn.commit()
    return jsonify({'message': 'Tâche ajoutée', 'id': task['id']}), 201

@app.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    with get_db() as conn:
        conn.execute('DELETE FROM task_resources WHERE taskId = ?', (task_id,))
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()
    return jsonify({'message': 'Tâche supprimée'})

@app.route('/resources', methods=['GET'])
def get_resources():
    with get_db() as conn:
        resources = conn.execute('SELECT * FROM resources').fetchall()
        return jsonify([dict(resource) for resource in resources])

@app.route('/resources', methods=['POST'])
def add_resource():
    data = request.get_json()
    resource = {
        'id': data['id'],
        'name': data['name'],
        'type': data['type'],
        'isAvailable': 1 if data['isAvailable'] else 0,
        'cost': data['cost']
    }
    with get_db() as conn:
        conn.execute('''
            INSERT INTO resources (id, name, type, isAvailable, cost)
            VALUES (:id, :name, :type, :isAvailable, :cost)
        ''', resource)
        conn.commit()
    return jsonify({'message': 'Ressource ajoutée', 'id': resource['id']}), 201

@app.route('/resources/<resource_id>', methods=['DELETE'])
def delete_resource(resource_id):
    with get_db() as conn:
        conn.execute('DELETE FROM task_resources WHERE resourceId = ?', (resource_id,))
        conn.execute('DELETE FROM resources WHERE id = ?', (resource_id,))
        conn.commit()
    return jsonify({'message': 'Ressource supprimée'})

@app.route('/optimize', methods=['POST'])
def optimize_schedule():
    data = request.get_json()
    tasks = data.get('tasks', [])
    resources = data.get('resources', [])
    if not tasks:
        return jsonify({'error': 'Aucune tâche fournie'}), 400

    prob = LpProblem("Minimize_Makespan", LpMinimize)
    start_times = {t['id']: LpVariable(f"start_{t['id']}", 0) for t in tasks}
    makespan = LpVariable("makespan", 0)
    prob += makespan

    now = datetime.datetime.now()
    for task in tasks:
        deadline = datetime.datetime.fromisoformat(task['deadline'])
        prob += start_times[task['id']] + task['duration'] <= (deadline - now).total_seconds() / 3600
        if task['dependencies']:
            for dep_id in task['dependencies']:
                if dep_id:
                    dep_task = next((t for t in tasks if t['id'] == dep_id), None)
                    if dep_task:
                        prob += start_times[task['id']] >= start_times[dep_id] + dep_task['duration']
        prob += makespan >= start_times[task['id']] + task['duration']

    resource_tasks = {}
    for task in tasks:
        for res_id in task.get('resources', []):
            resource_tasks.setdefault(res_id, []).append(task['id'])

    for res_id, task_ids in resource_tasks.items():
        if len(task_ids) > 1:
            for i in range(len(task_ids)):
                for j in range(i + 1, len(task_ids)):
                    t1 = next(t for t in tasks if t['id'] == task_ids[i])
                    t2 = next(t for t in tasks if t['id'] == task_ids[j])
                    b = LpVariable(f"b_{t1['id']}_{t2['id']}", 0, 1, cat='Binary')
                    M = 1000
                    prob += start_times[t1['id']] + t1['duration'] <= start_times[t2['id']] + M * (1 - b)
                    prob += start_times[t2['id']] + t2['duration'] <= start_times[t1['id']] + M * b

    prob.solve()
    if prob.status != 1:  # 1 = Optimal
        return jsonify({'error': 'Aucune solution optimale trouvée'}), 500

    optimized_tasks = []
    for task in tasks:
        start_time = now + datetime.timedelta(hours=value(start_times[task['id']]))
        task['startTime'] = start_time.isoformat()
        optimized_tasks.append(task)

    throughput = len(tasks) / value(makespan) if value(makespan) > 0 else 0

    with get_db() as conn:
        conn.execute('DELETE FROM tasks')
        conn.execute('DELETE FROM task_resources')
        for task in optimized_tasks:
            conn.execute('''
                INSERT INTO tasks (id, name, description, deadline, duration, priority, startTime, dependencies)
                VALUES (:id, :name, :description, :deadline, :duration, :priority, :startTime, :dependencies)
            ''', {
                'id': task['id'],
                'name': task['name'],
                'description': task['description'],
                'deadline': task['deadline'],
                'duration': task['duration'],
                'priority': task['priority'],
                'startTime': task['startTime'],
                'dependencies': ','.join(task['dependencies'])
            })
            for res_id in task.get('resources', []):
                conn.execute('INSERT INTO task_resources (taskId, resourceId) VALUES (?, ?)', (task['id'], res_id))
        conn.commit()

    return jsonify({
        'makespan': value(makespan),
        'throughput': throughput,
        'tasks': optimized_tasks
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)