import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() {
  runApp(const TaskPlanningApp());
}

class Task {
  String id;
  String name;
  String description;
  DateTime deadline;
  int duration;
  int priority;
  List<String> dependencies;
  List<String> assignedResources;
  DateTime? startTime;

  Task({
    required this.id,
    required this.name,
    required this.description,
    required this.deadline,
    required this.duration,
    required this.priority,
    this.dependencies = const [],
    this.assignedResources = const [],
    this.startTime,
  });

  factory Task.fromJson(Map<String, dynamic> json) {
    return Task(
      id: json['id'],
      name: json['name'],
      description: json['description'],
      deadline: DateTime.parse(json['deadline']),
      duration: json['duration'],
      priority: json['priority'],
      dependencies: (json['dependencies'] as String? ?? '').split(',').where((d) => d.isNotEmpty).toList(),
      assignedResources: json['resources'] != null ? List<String>.from(json['resources']) : [],
      startTime: json['startTime'] != null ? DateTime.parse(json['startTime']) : null,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'description': description,
        'deadline': deadline.toIso8601String(),
        'duration': duration,
        'priority': priority,
        'dependencies': dependencies,
        'resources': assignedResources,
      };
}

class Resource {
  String id;
  String name;
  String type;
  bool isAvailable;
  double cost;

  Resource({
    required this.id,
    required this.name,
    required this.type,
    this.isAvailable = true,
    required this.cost,
  });

  factory Resource.fromJson(Map<String, dynamic> json) => Resource(
        id: json['id'],
        name: json['name'],
        type: json['type'],
        isAvailable: json['isAvailable'] == 1,
        cost: json['cost'].toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'type': type,
        'isAvailable': isAvailable,
        'cost': cost,
      };
}

class TaskPlanningApp extends StatelessWidget {
  const TaskPlanningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Planification des Tâches',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;
  List<Task> _tasks = [];
  List<Resource> _resources = [];
  double _makespan = 0;
  double _throughput = 0;

  @override
  void initState() {
    super.initState();
    _fetchTasks();
    _fetchResources();
  }

  Future<void> _fetchTasks() async {
    final response = await http.get(Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/tasks'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      setState(() {
        _tasks = data.map((t) => Task.fromJson(t)).toList();
      });
    }
  }

  Future<void> _fetchResources() async {
    final response = await http.get(Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/resources'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      setState(() {
        _resources = data.map((r) => Resource.fromJson(r)).toList();
      });
    }
  }

  void _addTask(Task task) async {
    final response = await http.post(
      Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/tasks'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(task.toJson()),
    );
    if (response.statusCode == 201) await _fetchTasks();
  }

  void _addResource(Resource resource) async {
    final response = await http.post(
      Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/resources'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(resource.toJson()),
    );
    if (response.statusCode == 201) await _fetchResources();
  }

  void _deleteTask(String taskId) async {
    final response = await http.delete(Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/tasks/$taskId'));
    if (response.statusCode == 200) await _fetchTasks();
  }

  void _deleteResource(String resourceId) async {
    final response = await http.delete(Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/resources/$resourceId'));
    if (response.statusCode == 200) await _fetchResources();
  }

  Future<void> _planTasks() async {
    final response = await http.post(
      Uri.parse('https://redesigned-dollop-975vrgvx6q462xqwq-3000.app.github.dev/optimize'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'tasks': _tasks.map((t) => t.toJson()).toList(),
        'resources': _resources.map((r) => r.toJson()).toList(),
      }),
    );
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      print("Response from /optimize: $data");  // Debug: voir la réponse brute
      setState(() {
        _makespan = data['makespan'].toDouble();
        _throughput = data['throughput'].toDouble();
        _tasks = (data['tasks'] as List).map((t) => Task.fromJson(t)).toList();
        print("Updated tasks: ${_tasks.map((t) => {'id': t.id, 'startTime': t.startTime})}");  // Debug: vérifier startTime
      });
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Planning généré !')));
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(jsonDecode(response.body)['error'] ?? 'Erreur lors de la planification')),
      );
    }
  }

  void _showAddTaskDialog() {
    final nameController = TextEditingController();
    final descController = TextEditingController();
    final selectedDependencies = <String>[];
    final selectedResources = <String>[];
    int priority = 1;
    int duration = 1;
    DateTime deadline = DateTime.now().add(const Duration(days: 1));
    TimeOfDay time = TimeOfDay.now();

    Future<void> selectDate(BuildContext context) async {
      final pickedDate = await showDatePicker(
        context: context,
        initialDate: deadline,
        firstDate: DateTime.now(),
        lastDate: DateTime(2100),
      );
      if (pickedDate != null) {
        final pickedTime = await showTimePicker(context: context, initialTime: time);
        if (pickedTime != null) {
          setState(() {
            deadline = DateTime(pickedDate.year, pickedDate.month, pickedDate.day, pickedTime.hour, pickedTime.minute);
            time = pickedTime;
          });
        }
      }
    }

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('Nouvelle Tâche'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(controller: nameController, decoration: const InputDecoration(labelText: 'Nom')),
                TextField(controller: descController, decoration: const InputDecoration(labelText: 'Description')),
                Row(
                  children: [
                    const Text('Priorité: '),
                    Expanded(
                      child: Slider(
                        value: priority.toDouble(),
                        min: 1,
                        max: 3,
                        divisions: 2,
                        label: priority.toString(),
                        onChanged: (value) => setState(() => priority = value.round()),
                      ),
                    ),
                  ],
                ),
                Row(
                  children: [
                    const Text('Durée (h): '),
                    Expanded(
                      child: Slider(
                        value: duration.toDouble(),
                        min: 1,
                        max: 24,
                        divisions: 23,
                        label: duration.toString(),
                        onChanged: (value) => setState(() => duration = value.round()),
                      ),
                    ),
                  ],
                ),
                Row(
                  children: [
                    const Text('Deadline: '),
                    TextButton(
                      onPressed: () => selectDate(context),
                      child: Text('${deadline.day}/${deadline.month} ${time.format(context)}'),
                    ),
                  ],
                ),
                if (_tasks.isNotEmpty) ...[
                  const Text('Dépendances:'),
                  ..._tasks.map((t) => CheckboxListTile(
                        title: Text(t.name),
                        value: selectedDependencies.contains(t.id),
                        onChanged: (val) => setState(() {
                          if (val!) selectedDependencies.add(t.id);
                          else selectedDependencies.remove(t.id);
                        }),
                      )),
                ],
                if (_resources.isNotEmpty) ...[
                  const Text('Ressources:'),
                  ..._resources.map((r) => CheckboxListTile(
                        title: Text(r.name),
                        value: selectedResources.contains(r.id),
                        onChanged: (val) => setState(() {
                          if (val!) selectedResources.add(r.id);
                          else selectedResources.remove(r.id);
                        }),
                      )),
                ],
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Annuler')),
            ElevatedButton(
              onPressed: () {
                _addTask(Task(
                  id: DateTime.now().millisecondsSinceEpoch.toString(),
                  name: nameController.text,
                  description: descController.text,
                  deadline: deadline,
                  duration: duration,
                  priority: priority,
                  dependencies: selectedDependencies,
                  assignedResources: selectedResources,
                ));
                Navigator.pop(context);
              },
              child: const Text('Ajouter'),
            ),
          ],
        ),
      ),
    );
  }

  void _showAddResourceDialog() {
    final nameController = TextEditingController();
    final costController = TextEditingController();
    String type = 'Personnel';
    bool isAvailable = true;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('Nouvelle Ressource'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: nameController, decoration: const InputDecoration(labelText: 'Nom')),
              DropdownButton<String>(
                value: type,
                items: ['Personnel', 'Machine', 'Serveur'].map((t) => DropdownMenuItem(value: t, child: Text(t))).toList(),
                onChanged: (val) => setState(() => type = val!),
              ),
              SwitchListTile(
                title: const Text('Disponible'),
                value: isAvailable,
                onChanged: (val) => setState(() => isAvailable = val),
              ),
              TextField(
                controller: costController,
                decoration: const InputDecoration(labelText: 'Coût'),
                keyboardType: TextInputType.number,
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Annuler')),
            ElevatedButton(
              onPressed: () {
                _addResource(Resource(
                  id: DateTime.now().millisecondsSinceEpoch.toString(),
                  name: nameController.text,
                  type: type,
                  isAvailable: isAvailable,
                  cost: double.tryParse(costController.text) ?? 0.0,
                ));
                Navigator.pop(context);
              },
              child: const Text('Ajouter'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Planification des Tâches'),
        actions: [
          IconButton(
            icon: const Icon(Icons.schedule),
            onPressed: _planTasks,
            tooltip: 'Planifier',
          ),
        ],
      ),
      body: IndexedStack(
        index: _selectedIndex,
        children: [
          ListView.builder(
            itemCount: _tasks.length,
            itemBuilder: (context, index) => ListTile(
              title: Text(_tasks[index].name),
              subtitle: Text('Durée: ${_tasks[index].duration}h | Deadline: ${_tasks[index].deadline.day}/${_tasks[index].deadline.month}'),
              trailing: IconButton(
                icon: const Icon(Icons.delete, color: Colors.red),
                onPressed: () => _deleteTask(_tasks[index].id),
              ),
            ),
          ),
          ListView.builder(
            itemCount: _resources.length,
            itemBuilder: (context, index) => ListTile(
              title: Text(_resources[index].name),
              subtitle: Text('Type: ${_resources[index].type} | Coût: ${_resources[index].cost}'),
              trailing: IconButton(
                icon: const Icon(Icons.delete, color: Colors.red),
                onPressed: () => _deleteResource(_resources[index].id),
              ),
            ),
          ),
          Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Text('Makespan: ${_makespan.toStringAsFixed(2)}h'),
                    Text('Throughput: ${_throughput.toStringAsFixed(2)} tâches/h'),
                  ],
                ),
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: _tasks.length,
                  itemBuilder: (context, index) => ListTile(
                    title: Text(_tasks[index].name),
                    subtitle: Text(
                      'Début: ${_tasks[index].startTime != null ? _tasks[index].startTime!.toString() : "Non planifié"} | Durée: ${_tasks[index].duration}h',
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.task), label: 'Tâches'),
          BottomNavigationBarItem(icon: Icon(Icons.people), label: 'Ressources'),
          BottomNavigationBarItem(icon: Icon(Icons.calendar_today), label: 'Planning'),
        ],
      ),
      floatingActionButton: _selectedIndex < 2
          ? FloatingActionButton(
              onPressed: () => _selectedIndex == 0 ? _showAddTaskDialog() : _showAddResourceDialog(),
              child: const Icon(Icons.add),
            )
          : null,
    );
  }
}