import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:timeline_tile/timeline_tile.dart';
import 'dart:html' as html;
import 'package:intl/intl.dart';
import 'dart:typed_data';

// Entry point of the application
void main() {
  runApp(const TaskPlanningApp());
}

// Task model
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
    DateTime? startTimeValue;
    if (json['startTime'] != null) {
      try {
        String startTimeStr = json['startTime'] as String;
        if (startTimeStr.endsWith('Z')) {
          startTimeStr = startTimeStr.substring(0, startTimeStr.length - 1);
        }
        if (startTimeStr.contains('+')) {
          startTimeStr = startTimeStr.split('+')[0];
        }
        startTimeValue = DateTime.parse(startTimeStr);
      } catch (e) {
        startTimeValue = null;
      }
    }

    return Task(
      id: json['id'] as String,
      name: json['name'] as String,
      description: json['description'] as String? ?? '',
      deadline: DateTime.parse(json['deadline'] as String),
      duration: (json['duration'] as num).toInt(),
      priority: json['priority'] as int,
      dependencies: json['dependencies'] is String
          ? (json['dependencies'] as String).split(',').where((d) => d.isNotEmpty).toList()
          : (json['dependencies'] as List<dynamic>?)?.map((d) => d.toString()).toList() ?? [],
      assignedResources: json['resources'] is String
          ? jsonDecode(json['resources'] as String).cast<String>()
          : (json['resources'] as List<dynamic>?)?.map((r) => r.toString()).toList() ?? [],
      startTime: startTimeValue,
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
        'startTime': startTime?.toIso8601String(),
      };
}

// Resource model
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
        id: json['id'] as String,
        name: json['name'] as String,
        type: json['type'] as String,
        isAvailable: json['isAvailable'] == 1,
        cost: (json['cost'] as num).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'type': type,
        'isAvailable': isAvailable,
        'cost': cost,
      };
}

// Main application widget
class TaskPlanningApp extends StatelessWidget {
  const TaskPlanningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Task Planner',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        brightness: Brightness.light,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.teal,
        brightness: Brightness.dark,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      themeMode: ThemeMode.system,
      home: const HomeScreen(),
    );
  }
}

// Home screen stateful widget
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;
  List<Task> _tasks = [];
  List<Task> _filteredTasks = [];
  List<Resource> _resources = [];
  double _makespan = 0;
  double _throughput = 0;
  Map<String, double> _penalties = {};
  bool _isLoading = false;
  String _optimizationMethod = 'PuLP';
  final String _baseUrl = 'https://flutter-task-planner-3.onrender.com';

  String _searchQuery = '';
  int? _priorityFilter;
  String? _resourceFilter;
  bool? _completedFilter;
  bool _sortAscending = true;
  bool _showFilters = false;

  @override
  void initState() {
    super.initState();
    _optimizationMethod = 'PuLP';
    _initializeExampleData();
    _applyFilters();
    _saveToCache();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _showWelcomeMessage();
      _showWelcomeDialog();
    });
  }

  // Show welcome SnackBar
  void _showWelcomeMessage() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Welcome! Add tasks or plan the example.'),
        backgroundColor: Colors.teal,
        duration: Duration(seconds: 5),
      ),
    );
  }

  // Show welcome dialog on first launch
  Future<void> _showWelcomeDialog() async {
    final prefs = await SharedPreferences.getInstance();
    bool hasSeenWelcome = prefs.getBool('hasSeenWelcome') ?? false;
    if (!hasSeenWelcome && mounted) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Welcome to Task Planner!'),
          content: const Text(
              'This application helps you plan tasks.\n'
              '- Try the example with 3 tasks (Step 1, 2, 3).\n'
              '- Add your own tasks with the + button.\n'
              '- Click "Plan" to generate a schedule.'),
          actions: [
            TextButton(
              onPressed: () {
                prefs.setBool('hasSeenWelcome', true);
                Navigator.pop(context);
              },
              child: const Text('Got it'),
            ),
          ],
        ),
      );
    }
  }

  // Initialize example data with tasks and resources
  void _initializeExampleData() {
    if (_tasks.isNotEmpty || _resources.isNotEmpty) return;

    DateTime now = DateTime.now();
    _resources = [
      Resource(
        id: 'R1',
        name: 'Team',
        type: 'Personnel',
        isAvailable: true,
        cost: 50.0,
      ),
      Resource(
        id: 'R2',
        name: 'Server',
        type: 'Server',
        isAvailable: true,
        cost: 20.0,
      ),
      Resource(
        id: 'R3',
        name: 'Machine',
        type: 'Machine',
        isAvailable: true,
        cost: 30.0,
      ),
    ];

    _tasks = [
      Task(
        id: 'T1',
        name: 'Step 1',
        description: 'First step of the process',
        deadline: now.add(const Duration(hours: 12)),
        duration: 4,
        priority: 1,
        dependencies: [],
        assignedResources: ['R1'],
      ),
      Task(
        id: 'T2',
        name: 'Step 2',
        description: 'Second step depending on Step 1',
        deadline: now.add(const Duration(hours: 18)),
        duration: 6,
        priority: 2,
        dependencies: ['T1'],
        assignedResources: ['R2'],
      ),
      Task(
        id: 'T3',
        name: 'Step 3',
        description: 'Final step using multiple resources',
        deadline: now.add(const Duration(hours: 24)),
        duration: 5,
        priority: 3,
        dependencies: ['T2'],
        assignedResources: ['R1', 'R3'],
      ),
    ];
  }

  // Reset to example data
  void _resetToExample() async {
    setState(() {
      _tasks.clear();
      _resources.clear();
      _initializeExampleData();
      _applyFilters();
    });
    await _saveToCache();
    _showSuccess('Example reset successfully');
  }

  // Load cached tasks and resources
  Future<void> _loadFromCache() async {
    final prefs = await SharedPreferences.getInstance();
    final cachedTasks = prefs.getString('tasks');
    final cachedResources = prefs.getString('resources');
    if (cachedTasks != null) {
      setState(() {
        _tasks = (jsonDecode(cachedTasks) as List).map((t) => Task.fromJson(t)).toList();
        _applyFilters();
      });
    }
    if (cachedResources != null) {
      setState(() {
        _resources = (jsonDecode(cachedResources) as List).map((r) => Resource.fromJson(r)).toList();
      });
    }
  }

  // Save tasks and resources to cache
  Future<void> _saveToCache() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('tasks', jsonEncode(_tasks.map((t) => t.toJson()).toList()));
    await prefs.setString('resources', jsonEncode(_resources.map((r) => r.toJson()).toList()));
  }

  // Fetch tasks from backend
  Future<void> _fetchTasks() async {
    setState(() => _isLoading = true);
    try {
      final response = await http.get(Uri.parse('$_baseUrl/tasks'));
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _tasks = data.map((t) => Task.fromJson(t)).toList();
          _applyFilters();
          _saveToCache();
        });
      } else {
        _showError('Error fetching tasks: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error in _fetchTasks: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Fetch resources from backend
  Future<void> _fetchResources() async {
    setState(() => _isLoading = true);
    try {
      final response = await http.get(Uri.parse('$_baseUrl/resources'));
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _resources = data.map((r) => Resource.fromJson(r)).toList();
          _saveToCache();
        });
      } else {
        _showError('Error fetching resources: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error in _fetchResources: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Add a new task
  Future<void> _addTask(Task task) async {
    if (!_validateTask(task, _tasks)) {
      return;
    }

    setState(() => _isLoading = true); // Afficher l'indicateur de chargement
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/tasks'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(task.toJson()),
      );

      if (response.statusCode == 201) {
        // Ne pas ajouter la tâche localement avant le fetch
        setState(() {
          _tasks.add(task);
          _applyFilters();
        });

        // Ensuite, mettre à jour avec les données du serveur
        await _fetchTasks();
        _showSuccess('Task added successfully');
      } else {
        _showError('Error adding task: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error in _addTask: $e');
    } finally {
      setState(() => _isLoading = false); // Cacher l'indicateur de chargement
    }

  
  }
  

  // Add a new resource
  Future<void> _addResource(Resource resource) async {
    setState(() => _isLoading = true);
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/resources'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(resource.toJson()),
      );
      if (response.statusCode == 201) {
        await _fetchResources();
        _showSuccess('Resource added successfully');
      } else {
        _showError('Error adding resource: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Update an existing task
  Future<void> _updateTask(Task task) async {
    if (!_validateTask(task, _tasks)) {
      return;
    }
    setState(() => _isLoading = true);
    try {
      await http.delete(Uri.parse('$_baseUrl/tasks/${task.id}'));
      await _addTask(task);
      await _fetchTasks();
      _showSuccess('Task updated successfully');
    } catch (e) {
      _showError('Error updating task: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Update an existing resource
  Future<void> _updateResource(Resource resource) async {
    setState(() => _isLoading = true);
    try {
      await http.delete(Uri.parse('$_baseUrl/resources/${resource.id}'));
      await _addResource(resource);
      await _fetchResources();
      _showSuccess('Resource updated successfully');
    } catch (e) {
      _showError('Error updating resource: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Delete a task
  Future<void> _deleteTask(String taskId) async {
    final bool? confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Confirm Deletion'),
        content: const Text('Are you sure you want to delete this task?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    setState(() => _isLoading = true);
    try {
      final response = await http.delete(Uri.parse('$_baseUrl/tasks/$taskId'));
      if (response.statusCode == 200) {
        await _fetchTasks();
        _showSuccess('Task deleted successfully');
      } else {
        _showError('Error deleting task: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Delete a resource
  Future<void> _deleteResource(String resourceId) async {
    final bool? confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Confirm Deletion'),
        content: const Text('Are you sure you want to delete this resource?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    setState(() => _isLoading = true);
    try {
      final response = await http.delete(Uri.parse('$_baseUrl/resources/$resourceId'));
      if (response.statusCode == 200) {
        await _fetchResources();
        _showSuccess('Resource deleted successfully');
      } else {
        _showError('Error deleting resource: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Network error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Check task feasibility
  void _checkFeasibility() {
    DateTime now = DateTime.now();
    for (var task in _tasks) {
      double totalDuration = task.duration.toDouble();
      for (var depId in task.dependencies) {
        var dep = _tasks.firstWhere(
            (t) => t.id == depId,
            orElse: () => Task(id: '', name: '', description: '', deadline: now, duration: 0, priority: 1));
        totalDuration += dep.duration;
      }
      if (task.deadline.difference(now).inHours < totalDuration) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text(
                  "Deadline for '${task.name}' too short, suggest ${DateFormat('MM/dd HH:mm').format(now.add(Duration(hours: totalDuration.ceil())))}?")),
        );
      }
    }
  }

  // Plan tasks using optimization
  Future<void> _planTasks() async {
    if (_tasks.isEmpty) {
      _showError('No tasks to plan');
      return;
    }
    setState(() => _isLoading = true);
    _checkFeasibility();
    if (!_validateAllTasks(_tasks)) {
      setState(() => _isLoading = false);
      return;
    }
    try {
      print('Sending POST to $_baseUrl/optimize with method: $_optimizationMethod');
      final response = await http.post(
        Uri.parse('$_baseUrl/optimize'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'tasks': _tasks.map((t) => t.toJson()).toList(),
          'resources': _resources.map((r) => r.toJson()).toList(),
          'method': _optimizationMethod,
        }),
      ).timeout(const Duration(seconds: 5), onTimeout: () {
        throw Exception('Timeout while calling /optimize');
      });

      print('Optimize response: ${response.statusCode} - ${response.body}');
      if (response.statusCode != 202) {
        throw Exception('Unexpected status code from /optimize: ${response.statusCode}');
      }

      final initialData = jsonDecode(response.body) as Map<String, dynamic>;
      if (initialData['status'] != 'pending' || initialData['id'] == null) {
        throw Exception('Invalid response from /optimize: $initialData');
      }

      final resultId = initialData['id'].toString();
      print('Starting polling for result_id: $resultId');

      const maxAttempts = 10;
      const pollingInterval = Duration(seconds: 1);
      Map<String, dynamic>? resultData;
      for (int i = 0; i < maxAttempts; i++) {
        print('Polling attempt ${i + 1} for $_baseUrl/result/$resultId');
        final resultResponse = await http.get(
          Uri.parse('$_baseUrl/result/$resultId'),
        ).timeout(const Duration(seconds: 5), onTimeout: () {
          throw Exception('Timeout while polling /result/$resultId');
        });

        print('Result response: ${resultResponse.statusCode} - ${resultResponse.body}');
        if (resultResponse.statusCode != 200) {
          throw Exception('Unexpected status code from /result/$resultId: ${resultResponse.statusCode}');
        }

        resultData = jsonDecode(resultResponse.body) as Map<String, dynamic>;
        print('Result data: $resultData');
        if (resultData['status'] == 'completed') {
          break;
        } else if (resultData['status'] == 'error') {
          throw Exception('Optimization failed: ${resultData['error'] ?? 'Unknown error'}');
        }
        await Future.delayed(pollingInterval);
      }

      if (resultData != null && resultData['status'] == 'completed') {
        final completedData = resultData;
        final optimizedTasks = (completedData['tasks'] as List<dynamic>?)?.map((t) => Task.fromJson(t as Map<String, dynamic>)).toList() ?? [];
        setState(() {
          _makespan = (completedData['makespan'] as num?)?.toDouble() ?? 0.0;
          _throughput = (completedData['throughput'] as num?)?.toDouble() ?? 0.0;
          _penalties = Map<String, double>.from(completedData['penalties'] as Map? ?? {});
          _tasks = optimizedTasks;
          _applyFilters();
          _saveToCache();
        });
        print('Planning completed, tasks updated: ${_tasks.map((t) => "${t.name}: ${t.startTime}").toList()}');
        _showSuccess('Schedule generated successfully!');
      } else {
        throw Exception('Failed to retrieve optimization result after $maxAttempts attempts');
      }
    } catch (e) {
      print('Planning error: $e');
      _showError('Planning error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Show error message
  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: Colors.red,
    ));
  }

  // Show success message
  void _showSuccess(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: Colors.green,
    ));
  }

  // Get dependency names for display
  String _getDependencyNames(List<String> dependencyIds) {
    return dependencyIds
        .map((id) => _tasks.firstWhere(
              (t) => t.id == id,
              orElse: () => Task(
                  id: id,
                  name: 'Unknown',
                  description: '',
                  deadline: DateTime.now(),
                  duration: 0,
                  priority: 1),
            ).name)
        .join(", ");
  }

  // Validate a task
  bool _validateTask(Task task, List<Task> allTasks) {
    if (task.startTime != null) {
      DateTime endTime = task.startTime!.add(Duration(hours: task.duration));
      if (endTime.isAfter(task.deadline)) {
        _showError(
            "Error: '${task.name}' - Duration (${task.duration}h) exceeds deadline");
        return false;
      }
    }

    for (String depId in task.dependencies) {
      Task? dependency = allTasks.firstWhere((t) => t.id == depId, orElse: () => null as Task);
      if (dependency != null && dependency.startTime != null) {
        DateTime depEndTime = dependency.startTime!.add(Duration(hours: dependency.duration));
        if (task.startTime != null && task.startTime!.isBefore(depEndTime)) {
          _showError(
              "Error: '${task.name}' - Start (${task.startTime}) before end of '${dependency.name}' (${depEndTime})");
          return false;
        }
      }
    }

    return true;
  }

  // Validate all tasks
  bool _validateAllTasks(List<Task> tasks) {
    for (Task task in tasks) {
      if (!_validateTask(task, tasks)) {
        return false;
      }
    }
    return true;
  }

  // Apply filters to tasks
  void _applyFilters() {
    setState(() {
      _filteredTasks = _tasks.where((task) {
        bool matchesSearch = _searchQuery.isEmpty ||
            task.name.toLowerCase().contains(_searchQuery.toLowerCase()) ||
            task.description.toLowerCase().contains(_searchQuery.toLowerCase());
        bool matchesPriority = _priorityFilter == null || task.priority == _priorityFilter;
        bool matchesResource = _resourceFilter == null ||
            task.assignedResources.contains(_resourceFilter);
        bool matchesCompleted = _completedFilter == null ||
            (_completedFilter == true && task.startTime != null) ||
            (_completedFilter == false && task.startTime == null);
        return matchesSearch && matchesPriority && matchesResource && matchesCompleted;
      }).toList();
    });
  }

  // Sort tasks by start time
  void _sortTasksByStartTime() {
    setState(() {
      _sortAscending = !_sortAscending;
      final tasksWithStartTime = _tasks.where((task) => task.startTime != null).toList();
      final tasksWithoutStartTime = _tasks.where((task) => task.startTime == null).toList();
      tasksWithStartTime.sort((a, b) {
        if (_sortAscending) {
          return a.startTime!.compareTo(b.startTime!);
        } else {
          return b.startTime!.compareTo(a.startTime!);
        }
      });
      _tasks = [...tasksWithStartTime, ...tasksWithoutStartTime];
      _applyFilters();
      _showSuccess('Tasks sorted by start time ${_sortAscending ? "ascending" : "descending"}');
    });
  }

  // Show dialog to add a new task
  void _showAddTaskDialog() {
    final nameController = TextEditingController();
    final descController = TextEditingController();
    final durationController = TextEditingController();
    final priorityController = TextEditingController();
    final selectedDependencies = <String>[];
    final selectedResources = <String>[];
    DateTime deadline = DateTime.now().add(const Duration(days: 1));
    TimeOfDay deadlineTime = TimeOfDay.now();
    DateTime? startTime;
    TimeOfDay? startTimeOfDay;

    Future<void> selectDate(BuildContext context, {bool isStartTime = false}) async {
      final pickedDate = await showDatePicker(
        context: context,
        initialDate: isStartTime ? (startTime ?? DateTime.now()) : deadline,
        firstDate: DateTime.now(),
        lastDate: DateTime(2100),
      );
      if (pickedDate != null) {
        final pickedTime = await showTimePicker(
          context: context,
          initialTime: isStartTime ? (startTimeOfDay ?? TimeOfDay.now()) : deadlineTime,
        );
        if (pickedTime != null) {
          if (isStartTime) {
            startTime = DateTime(pickedDate.year, pickedDate.month, pickedDate.day, pickedTime.hour, pickedTime.minute);
            startTimeOfDay = pickedTime;
          } else {
            deadline = DateTime(pickedDate.year, pickedDate.month, pickedDate.day, pickedTime.hour, pickedTime.minute);
            deadlineTime = pickedTime;
          }
        }
      }
    }

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) {
          return AlertDialog(
            title: const Text('New Task'),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                    controller: nameController,
                    decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: descController,
                    decoration: const InputDecoration(labelText: 'Description', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: durationController,
                    decoration: const InputDecoration(
                      labelText: 'Duration (hours)',
                      border: OutlineInputBorder(),
                      hintText: 'Integer, e.g., 3',
                    ),
                    keyboardType: TextInputType.number,
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: priorityController,
                    decoration: const InputDecoration(
                      labelText: 'Priority',
                      border: OutlineInputBorder(),
                      hintText: 'Integer, e.g., 1, 2, 3',
                    ),
                    keyboardType: TextInputType.number,
                  ),
                  Row(
                    children: [
                      const Text('Start: '),
                      TextButton(
                        onPressed: () => selectDate(context, isStartTime: true).then((_) => setState(() {})),
                        child: Text(startTime != null
                            ? '${startTime!.month}/${startTime!.day} ${startTimeOfDay!.format(context)}'
                            : 'Not set'),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      const Text('Deadline: '),
                      TextButton(
                        onPressed: () => selectDate(context).then((_) => setState(() {})),
                        child: Text('${deadline.month}/${deadline.day} ${deadlineTime.format(context)}'),
                      ),
                    ],
                  ),
                  if (_tasks.isNotEmpty) ...[
                    const Text('Dependencies:', style: TextStyle(fontWeight: FontWeight.bold)),
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
                    const Text('Resources:', style: TextStyle(fontWeight: FontWeight.bold)),
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
              TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
              ElevatedButton(
                onPressed: () async {
                  int? duration = int.tryParse(durationController.text);
                  int? priority = int.tryParse(priorityController.text);
                  if (nameController.text.isEmpty) {
                    _showError('Task name is required');
                    return;
                  }
                  if (duration == null || duration <= 0) {
                    _showError('Please enter a valid duration (integer > 0)');
                    return;
                  }
                  if (priority == null) {
                    _showError('Please enter a valid priority (integer)');
                    return;
                  }
                  final newTask = Task(
                    id: DateTime.now().millisecondsSinceEpoch.toString(),
                    name: nameController.text,
                    description: descController.text,
                    deadline: deadline,
                    duration: duration,
                    priority: priority,
                    dependencies: selectedDependencies,
                    assignedResources: selectedResources,
                    startTime: startTime,
                  );
                  if (_validateTask(newTask, _tasks)) {
                    Navigator.pop(context);
                    await _addTask(newTask);
                  }
                },
                child: const Text('Add'),
              ),
            ],
          );
        },
      ),
    );
  }

  // Show dialog to edit a task
  void _showEditTaskDialog(Task task) {
    final nameController = TextEditingController(text: task.name);
    final descController = TextEditingController(text: task.description);
    final durationController = TextEditingController(text: task.duration.toString());
    final priorityController = TextEditingController(text: task.priority.toString());
    final selectedDependencies = task.dependencies.toList();
    final selectedResources = task.assignedResources.toList();
    DateTime deadline = task.deadline;
    TimeOfDay deadlineTime = TimeOfDay.fromDateTime(task.deadline);
    DateTime? startTime = task.startTime;
    TimeOfDay? startTimeOfDay = startTime != null ? TimeOfDay.fromDateTime(startTime) : null;

    Future<void> selectDate(BuildContext context, {bool isStartTime = false}) async {
      final pickedDate = await showDatePicker(
        context: context,
        initialDate: isStartTime ? (startTime ?? DateTime.now()) : deadline,
        firstDate: DateTime.now(),
        lastDate: DateTime(2100),
      );
      if (pickedDate != null) {
        final pickedTime = await showTimePicker(
          context: context,
          initialTime: isStartTime ? (startTimeOfDay ?? TimeOfDay.now()) : deadlineTime,
        );
        if (pickedTime != null) {
          if (isStartTime) {
            startTime = DateTime(pickedDate.year, pickedDate.month, pickedDate.day, pickedTime.hour, pickedTime.minute);
            startTimeOfDay = pickedTime;
          } else {
            deadline = DateTime(pickedDate.year, pickedDate.month, pickedDate.day, pickedTime.hour, pickedTime.minute);
            deadlineTime = pickedTime;
          }
        }
      }
    }

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) {
          return AlertDialog(
            title: const Text('Edit Task'),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                    controller: nameController,
                    decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: descController,
                    decoration: const InputDecoration(labelText: 'Description', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: durationController,
                    decoration: const InputDecoration(
                      labelText: 'Duration (hours)',
                      border: OutlineInputBorder(),
                      hintText: 'Integer, e.g., 3',
                    ),
                    keyboardType: TextInputType.number,
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: priorityController,
                    decoration: const InputDecoration(
                      labelText: 'Priority',
                      border: OutlineInputBorder(),
                      hintText: 'Integer, e.g., 1, 2, 3',
                    ),
                    keyboardType: TextInputType.number,
                  ),
                  Row(
                    children: [
                      const Text('Start: '),
                      TextButton(
                        onPressed: () => selectDate(context, isStartTime: true).then((_) => setState(() {})),
                        child: Text(startTime != null
                            ? '${startTime!.month}/${startTime!.day} ${startTimeOfDay!.format(context)}'
                            : 'Not set'),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      const Text('Deadline: '),
                      TextButton(
                        onPressed: () => selectDate(context).then((_) => setState(() {})),
                        child: Text('${deadline.month}/${deadline.day} ${deadlineTime.format(context)}'),
                      ),
                    ],
                  ),
                  if (_tasks.isNotEmpty) ...[
                    const Text('Dependencies:', style: TextStyle(fontWeight: FontWeight.bold)),
                    ..._tasks.where((t) => t.id != task.id).map((t) => CheckboxListTile(
                          title: Text(t.name),
                          value: selectedDependencies.contains(t.id),
                          onChanged: (val) => setState(() {
                            if (val!) selectedDependencies.add(t.id);
                            else selectedDependencies.remove(t.id);
                          }),
                        )),
                  ],
                  if (_resources.isNotEmpty) ...[
                    const Text('Resources:', style: TextStyle(fontWeight: FontWeight.bold)),
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
              TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
              ElevatedButton(
                onPressed: () {
                  int? duration = int.tryParse(durationController.text);
                  int? priority = int.tryParse(priorityController.text);
                  if (nameController.text.isEmpty) {
                    _showError('Task name is required');
                    return;
                  }
                  if (duration == null || duration <= 0) {
                    _showError('Please enter a valid duration (integer > 0)');
                    return;
                  }
                  if (priority == null) {
                    _showError('Please enter a valid priority (integer)');
                    return;
                  }
                  final updatedTask = Task(
                    id: task.id,
                    name: nameController.text,
                    description: descController.text,
                    deadline: deadline,
                    duration: duration,
                    priority: priority,
                    dependencies: selectedDependencies,
                    assignedResources: selectedResources,
                    startTime: startTime,
                  );
                  if (_validateTask(updatedTask, _tasks)) {
                    Navigator.pop(context);
                    _updateTask(updatedTask);
                  }
                },
                child: const Text('Update'),
              ),
            ],
          );
        },
      ),
    );
  }

  // Show dialog to add a new resource
  void _showAddResourceDialog() {
  final nameController = TextEditingController();
  final costController = TextEditingController();
  String type = 'Personnel';
  bool isAvailable = true;

  showDialog(
    context: context,
    builder: (context) => StatefulBuilder(
      builder: (context, setState) {
        return AlertDialog(
          title: const Text('New Resource'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 10),
              DropdownButton<String>(
                value: type,
                items: ['Personnel', 'Machine', 'Server'].map((t) => DropdownMenuItem(value: t, child: Text(t))).toList(),
                onChanged: (val) => setState(() => type = val!),
              ),
              SwitchListTile(
                title: const Text('Available'),
                value: isAvailable,
                onChanged: (val) => setState(() => isAvailable = val),
              ),
              TextField(
                controller: costController,
                decoration: const InputDecoration(labelText: 'Cost', border: OutlineInputBorder()),
                keyboardType: TextInputType.number,
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            ElevatedButton(
              onPressed: () {
                if (nameController.text.isEmpty) {
                  _showError('Resource name is required');
                  return;
                }
                if (double.tryParse(costController.text) == null) {
                  _showError('Please enter a valid cost');
                  return;
                }
                final newResource = Resource(
                  id: DateTime.now().millisecondsSinceEpoch.toString(),
                  name: nameController.text,
                  type: type,
                  isAvailable: isAvailable,
                  cost: double.tryParse(costController.text) ?? 0.0,
                );
                Navigator.pop(context);
                _addResource(newResource);
              },
              child: const Text('Add'),
            ),
          ],
        );
      },
    ),
  );
 }


  // Show dialog to edit a resource
  void _showEditResourceDialog(Resource resource) {
  final nameController = TextEditingController(text: resource.name);
  final costController = TextEditingController(text: resource.cost.toString());
  String type = resource.type;
  bool isAvailable = resource.isAvailable;

  showDialog(
    context: context,
    builder: (context) => StatefulBuilder(
      builder: (context, setState) {
        return AlertDialog(
          title: const Text('Edit Resource'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 10),
              DropdownButton<String>(
                value: type,
                items: ['Personnel', 'Machine', 'Server'].map((t) => DropdownMenuItem(value: t, child: Text(t))).toList(),
                onChanged: (val) => setState(() => type = val!),
              ),
              SwitchListTile(
                title: const Text('Available'),
                value: isAvailable,
                onChanged: (val) => setState(() => isAvailable = val),
              ),
              TextField(
                controller: costController,
                decoration: const InputDecoration(labelText: 'Cost', border: OutlineInputBorder()),
                keyboardType: TextInputType.number,
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            ElevatedButton(
              onPressed: () {
                if (nameController.text.isEmpty) {
                  _showError('Resource name is required');
                  return;
                }
                if (double.tryParse(costController.text) == null) {
                  _showError('Please enter a valid cost');
                  return;
                }
                final updatedResource = Resource(
                  id: resource.id,
                  name: nameController.text,
                  type: type,
                  isAvailable: isAvailable,
                  cost: double.tryParse(costController.text) ?? 0.0,
                );
                Navigator.pop(context);
                _updateResource(updatedResource);
              },
              child: const Text('Update'),
            ),
          ],
        );
      },
    ),
  );
 }


  // Export planning to CSV
  Future<void> _exportPlanning() async {
    if (_tasks.isEmpty) {
      _showError('No tasks to export');
      return;
    }

    // Helper function to escape CSV fields
    String escapeCsvField(String? input) {
      if (input == null || input.isEmpty) return '""';
      // Escape quotes by doubling them and wrap the field in quotes
      final escaped = input.replaceAll('"', '""').replaceAll('\n', ' ').replaceAll('\r', '');
      return '"$escaped"';
    }

    final csv = StringBuffer();
    // Define column headers (in French for better compatibility)
    csv.writeln('ID;Nom;Description;Date limite;Durée (h);Priorité;Dépendances;Ressources;Début planifié');
    final dateFormat = DateFormat('dd/MM/yyyy HH:mm'); // French date format

    // Sort tasks by start time for better readability
    final sortedTasks = _tasks.toList()
      ..sort((a, b) {
        if (a.startTime == null && b.startTime == null) return 0;
        if (a.startTime == null) return 1;
        if (b.startTime == null) return -1;
        return a.startTime!.compareTo(b.startTime!);
      });

    for (final task in sortedTasks) {
      final dependencies = task.dependencies.isEmpty
          ? ''
          : task.dependencies.map((id) => _tasks.firstWhere((t) => t.id == id, orElse: () => Task(id: id, name: 'Unknown', description: '', deadline: DateTime.now(), duration: 0, priority: 1)).name).join(', ');
      final resources = task.assignedResources.isEmpty
          ? ''
          : task.assignedResources.map((id) => _resources.firstWhere((r) => r.id == id, orElse: () => Resource(id: id, name: 'Unknown', type: 'Unknown', cost: 0.0)).name).join(', ');
      final startTime = task.startTime != null ? dateFormat.format(task.startTime!) : '';
      final deadline = dateFormat.format(task.deadline);

      csv.writeln([
        escapeCsvField(task.id),
        escapeCsvField(task.name),
        escapeCsvField(task.description),
        escapeCsvField(deadline),
        escapeCsvField(task.duration.toString()),
        escapeCsvField(task.priority.toString()),
        escapeCsvField(dependencies),
        escapeCsvField(resources),
        escapeCsvField(startTime),
      ].join(';'));
    }

    // Add UTF-8 BOM for Excel compatibility
    final bom = [0xEF, 0xBB, 0xBF];
    final bytes = Uint8List.fromList(bom + utf8.encode(csv.toString()));
    final blob = html.Blob([bytes], 'text/csv;charset=utf-8');

    final url = html.Url.createObjectUrlFromBlob(blob);
    final anchor = html.AnchorElement(href: url)
      ..setAttribute('download', 'planning_${DateTime.now().millisecondsSinceEpoch}.csv')
      ..click();
    html.Url.revokeObjectUrl(url);

    _showSuccess('Planning exporté avec succès');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Task Planner'),
        actions: [
          if (_selectedIndex == 2)
            IconButton(
              icon: const Icon(Icons.download),
              onPressed: _exportPlanning,
              tooltip: 'Export to CSV',
            ),
          PopupMenuButton<String>(
            tooltip: 'Optimization Method',
            onSelected: (value) {
              setState(() {
                _optimizationMethod = value;
              });
              _showSuccess('Optimization method: $value');
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'PuLP',
                child: Text('PuLP (Exact)'),
              ),
              const PopupMenuItem(
                value: 'SimulatedAnnealing',
                child: Text('Simulated Annealing (Metaheuristic)'),
              ),
            ],
            child: Chip(
              label: Text(_optimizationMethod),
              avatar: const Icon(Icons.settings),
            ),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _buildResponsiveLayout(),
      bottomNavigationBar: _isMobile(context)
          ? BottomNavigationBar(
              currentIndex: _selectedIndex,
              onTap: (index) => setState(() => _selectedIndex = index),
              items: const [
                BottomNavigationBarItem(icon: Icon(Icons.task), label: 'Tasks'),
                BottomNavigationBarItem(icon: Icon(Icons.people), label: 'Resources'),
                BottomNavigationBarItem(icon: Icon(Icons.calendar_today), label: 'Schedule'),
              ],
            )
          : null,
      floatingActionButton: _selectedIndex < 2
          ? FloatingActionButton(
              onPressed: () => _selectedIndex == 0 ? _showAddTaskDialog() : _showAddResourceDialog(),
              child: const Icon(Icons.add),
              tooltip: _selectedIndex == 0 ? 'Add Task' : 'Add Resource',
            )
          : FloatingActionButton(
              onPressed: _planTasks,
              child: const Icon(Icons.play_arrow),
              tooltip: 'Generate Schedule',
            ),
    );
  }

  // Check if the device is mobile
  bool _isMobile(BuildContext context) => MediaQuery.of(context).size.width < 650;

  // Check if the device is a tablet
  bool _isTablet(BuildContext context) => MediaQuery.of(context).size.width >= 650 && MediaQuery.of(context).size.width < 1100;

  // Check if the device is a desktop
  bool _isDesktop(BuildContext context) => MediaQuery.of(context).size.width >= 1100;

  // Build responsive layout based on device type
  Widget _buildResponsiveLayout() {
    if (_isDesktop(context)) {
      return _buildDesktopView();
    } else if (_isTablet(context)) {
      return _buildTabletView();
    } else {
      return _buildMobileView();
    }
  }

  // Build mobile view
  Widget _buildMobileView() {
    return IndexedStack(
      index: _selectedIndex,
      children: [
        _buildTasksView(),
        _buildResourcesView(),
        _buildPlanningView(),
      ],
    );
  }

  // Build tablet view
  Widget _buildTabletView() {
    return Row(
      children: [
        NavigationRail(
          selectedIndex: _selectedIndex,
          onDestinationSelected: (index) => setState(() => _selectedIndex = index),
          labelType: NavigationRailLabelType.selected,
          destinations: const [
            NavigationRailDestination(
              icon: Icon(Icons.task),
              label: Text('Tasks'),
            ),
            NavigationRailDestination(
              icon: Icon(Icons.people),
              label: Text('Resources'),
            ),
            NavigationRailDestination(
              icon: Icon(Icons.calendar_today),
              label: Text('Schedule'),
            ),
          ],
        ),
        Expanded(
          child: IndexedStack(
            index: _selectedIndex,
            children: [
              _buildTasksView(),
              _buildResourcesView(),
              _buildPlanningView(),
            ],
          ),
        ),
      ],
    );
  }

  // Build desktop view
  Widget _buildDesktopView() {
    return Row(
      children: [
        NavigationRail(
          selectedIndex: _selectedIndex,
          onDestinationSelected: (index) => setState(() => _selectedIndex = index),
          extended: true,
          destinations: const [
            NavigationRailDestination(
              icon: Icon(Icons.task),
              label: Text('Tasks'),
            ),
            NavigationRailDestination(
              icon: Icon(Icons.people),
              label: Text('Resources'),
            ),
            NavigationRailDestination(
              icon: Icon(Icons.calendar_today),
              label: Text('Schedule'),
            ),
          ],
        ),
        Expanded(
          child: IndexedStack(
            index: _selectedIndex,
            children: [
              _buildTasksView(),
              _buildResourcesView(),
              _buildPlanningView(),
            ],
          ),
        ),
      ],
    );
  }

  // Build tasks view
  Widget _buildTasksView() {
    return Column(
      children: [
        _buildSearchFilterBar(),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              ElevatedButton.icon(
                onPressed: _resetToExample,
                icon: const Icon(Icons.refresh),
                label: const Text('Reset Example'),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
              ),
            ],
          ),
        ),
        Expanded(
          child: _filteredTasks.isEmpty
              ? const Center(child: Text('No tasks found. Use the + button to add a task.'))
              : ListView.builder(
                  itemCount: _filteredTasks.length,
                  itemBuilder: (context, index) {
                    final task = _filteredTasks[index];
                    final isExampleTask = ['T1', 'T2', 'T3'].contains(task.id);
                    return Tooltip(
                      message: 'Edit or delete this task',
                      child: Card(
                        margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        color: isExampleTask ? Colors.teal[50] : null,
                        child: ExpansionTile(
                          leading: CircleAvatar(
                              backgroundColor: Colors.teal,
                              child: Text(task.priority.toString(), style: const TextStyle(color: Colors.white))),
                          title: Text(task.name),
                          subtitle: Text('Duration: ${task.duration}h | Deadline: ${DateFormat('MM/dd HH:mm').format(task.deadline)}'),
                          children: [
                            Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  if (task.description.isNotEmpty) ...[
                                    const Text('Description:', style: TextStyle(fontWeight: FontWeight.bold)),
                                    Text(task.description),
                                    const SizedBox(height: 8),
                                  ],
                                  if (task.dependencies.isNotEmpty) ...[
                                    const Text('Dependencies:', style: TextStyle(fontWeight: FontWeight.bold)),
                                    Text(_getDependencyNames(task.dependencies)),
                                    const SizedBox(height: 8),
                                  ],
                                  if (task.assignedResources.isNotEmpty) ...[
                                    const Text('Resources:', style: TextStyle(fontWeight: FontWeight.bold)),
                                    Text(task.assignedResources.map((id) => _resources.firstWhere((r) => r.id == id).name).join(', ')),
                                    const SizedBox(height: 8),
                                  ],
                                  if (task.startTime != null) ...[
                                    const Text('Planned Start:', style: TextStyle(fontWeight: FontWeight.bold)),
                                    Text(DateFormat('MM/dd/yyyy HH:mm').format(task.startTime!)),
                                    const SizedBox(height: 8),
                                  ],
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.end,
                                    children: [
                                      TextButton.icon(
                                        icon: const Icon(Icons.edit),
                                        label: const Text('Edit'),
                                        onPressed: () => _showEditTaskDialog(task),
                                      ),
                                      TextButton.icon(
                                        icon: const Icon(Icons.delete),
                                        label: const Text('Delete'),
                                        onPressed: () => _deleteTask(task.id),
                                        style: TextButton.styleFrom(foregroundColor: Colors.red),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ],
                          trailing: task.startTime != null
                              ? const Icon(Icons.check_circle, color: Colors.green)
                              : const Icon(Icons.schedule, color: Colors.orange),
                        ),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }

  // Build search and filter bar
  Widget _buildSearchFilterBar() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    labelText: 'Search',
                    prefixIcon: const Icon(Icons.search),
                    suffixIcon: _searchQuery.isNotEmpty
                        ? IconButton(
                            icon: const Icon(Icons.clear),
                            onPressed: () {
                              setState(() {
                                _searchQuery = '';
                                _applyFilters();
                              });
                            },
                          )
                        : null,
                    border: const OutlineInputBorder(),
                    contentPadding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
                  ),
                  onChanged: (value) {
                    setState(() {
                      _searchQuery = value;
                      _applyFilters();
                    });
                  },
                ),
              ),
              IconButton(
                icon: Icon(_showFilters ? Icons.filter_list_off : Icons.filter_list),
                tooltip: _showFilters ? 'Hide Filters' : 'Show Filters',
                onPressed: () {
                  setState(() {
                    _showFilters = !_showFilters;
                  });
                },
              ),
            ],
          ),
          if (_showFilters)
            AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              height: _showFilters ? null : 0,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8.0),
                child: Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Filters',
                          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                        ),
                        const Divider(),
                        Wrap(
                          spacing: 8.0,
                          runSpacing: 8.0,
                          children: [
                            _buildPriorityFilter(),
                            _buildResourceFilter(),
                            _buildCompletedFilter(),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            TextButton(
                              onPressed: _resetFilters,
                              child: const Text('Reset'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  // Build priority filter dropdown
  Widget _buildPriorityFilter() {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 180),
      child: InputDecorator(
        decoration: const InputDecoration(
          labelText: 'Priority',
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<int?>(
            value: _priorityFilter,
            isDense: true,
            isExpanded: true,
            hint: const Text('All'),
            onChanged: (value) {
              setState(() {
                _priorityFilter = value;
                _applyFilters();
              });
            },
            items: [
              const DropdownMenuItem(value: null, child: Text('All')),
              ...List.generate(10, (index) => index + 1)
                  .map((p) => DropdownMenuItem(value: p, child: Text(p.toString())))
                  .toList(),
            ],
          ),
        ),
      ),
    );
  }

  // Build resource filter dropdown
  Widget _buildResourceFilter() {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 180),
      child: InputDecorator(
        decoration: const InputDecoration(
          labelText: 'Resource',
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<String?>(
            value: _resourceFilter,
            isDense: true,
            isExpanded: true,
            hint: const Text('All'),
            onChanged: (value) {
              setState(() {
                _resourceFilter = value;
                _applyFilters();
              });
            },
            items: [
              const DropdownMenuItem(value: null, child: Text('All')),
              ..._resources.map((resource) => DropdownMenuItem(
                    value: resource.id,
                    child: Text(resource.name),
                  )),
            ],
          ),
        ),
      ),
    );
  }

  // Build completed filter dropdown
  Widget _buildCompletedFilter() {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 180),
      child: InputDecorator(
        decoration: const InputDecoration(
          labelText: 'Status',
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<bool?>(
            value: _completedFilter,
            isDense: true,
            isExpanded: true,
            hint: const Text('All'),
            onChanged: (value) {
              setState(() {
                _completedFilter = value;
                _applyFilters();
              });
            },
            items: const [
              DropdownMenuItem(value: null, child: Text('All')),
              DropdownMenuItem(value: true, child: Text('Scheduled')),
              DropdownMenuItem(value: false, child: Text('Not Scheduled')),
            ],
          ),
        ),
      ),
    );
  }

  // Reset all filters
  void _resetFilters() {
    setState(() {
      _searchQuery = '';
      _priorityFilter = null;
      _resourceFilter = null;
      _completedFilter = null;
      _applyFilters();
    });
  }

  // Build resources view
  Widget _buildResourcesView() {
    return _resources.isEmpty
        ? const Center(child: Text('No resources added. Click + to start.'))
        : ListView.builder(
            itemCount: _resources.length,
            itemBuilder: (context, index) {
              final resource = _resources[index];
              return Tooltip(
                message: 'Resource: ${resource.type}',
                child: Card(
                  margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: resource.isAvailable ? Colors.green : Colors.red,
                      child: Icon(
                        resource.type == 'Personnel'
                            ? Icons.person
                            : resource.type == 'Machine'
                                ? Icons.precision_manufacturing
                                : Icons.computer,
                        color: Colors.white,
                      ),
                    ),
                    title: Text(resource.name),
                    subtitle: Text('Type: ${resource.type} | Cost: ${resource.cost}'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                            icon: const Icon(Icons.edit),
                            onPressed: () => _showEditResourceDialog(resource)),
                        IconButton(
                            icon: const Icon(Icons.delete),
                            onPressed: () => _deleteResource(resource.id)),
                      ],
                    ),
                  ),
                ),
              );
            },
          );
  }

  // Build planning view
  Widget _buildPlanningView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Makespan: ${_makespan.toStringAsFixed(2)}h',
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.teal),
                        ),
                        LinearProgressIndicator(
                            value: _makespan > 0 ? _makespan / 24 : 0, color: Colors.teal, minHeight: 8),
                        Text('Throughput: ${_throughput.toStringAsFixed(2)} tasks/h'),
                        LinearProgressIndicator(value: _throughput > 0 ? _throughput / 5 : 0, color: Colors.blue),
                      ],
                    ),
                  ),
                  Column(
                    children: [
                      IconButton(
                        icon: Icon(_sortAscending ? Icons.arrow_upward : Icons.arrow_downward),
                        onPressed: _sortTasksByStartTime,
                        tooltip: 'Sort by Start Time',
                        color: Colors.teal,
                      ),
                      const Text('Sort', style: TextStyle(fontSize: 12)),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ),
        Expanded(
          child: _tasks.isEmpty
              ? const Center(child: Text('No schedule. Add tasks and generate.'))
              : DefaultTabController(
                  length: 2,
                  child: Column(
                    children: [
                      const TabBar(
                        tabs: [
                          Tab(text: 'Timeline'),
                          Tab(text: 'Gantt'),
                        ],
                      ),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _buildTimelineView(),
                            _buildGanttView(),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ],
    );
  }

  // Build timeline view
  Widget _buildTimelineView() {
    return ListView.builder(
      itemCount: _tasks.length,
      itemBuilder: (context, index) {
        final task = _tasks[index];
        final dateFormat = DateFormat('HH:mm');

        return TimelineTile(
          alignment: TimelineAlign.manual,
          lineXY: 0.3,
          isFirst: index == 0,
          isLast: index == _tasks.length - 1,
          indicatorStyle: IndicatorStyle(
            width: 20,
            color: Colors.teal,
            padding: const EdgeInsets.all(4),
            iconStyle: task.startTime == null
                ? IconStyle(iconData: Icons.warning, color: Colors.white)
                : null,
          ),
          beforeLineStyle: LineStyle(color: Colors.teal, thickness: 2),
          startChild: GestureDetector(
            onTap: () => _showEditTaskDialog(task),
            child: Container(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                task.name,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          endChild: Container(
            padding: const EdgeInsets.all(8.0),
            constraints: const BoxConstraints(minWidth: 200),
            decoration: BoxDecoration(
              color: task.startTime == null ? Colors.grey[200] : Colors.white,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.teal),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  task.startTime != null
                      ? "${dateFormat.format(task.startTime!)} - ${dateFormat.format(task.startTime!.add(Duration(hours: task.duration)))}"
                      : "Not scheduled",
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                Text("Duration: ${task.duration}h"),
                Text("Deadline: ${dateFormat.format(task.deadline)}"),
                if (task.dependencies.isNotEmpty)
                  Text("Depends on: ${_getDependencyNames(task.dependencies)}",
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
              ],
            ),
          ),
        );
      },
    );
  }

  // Build Gantt view
  Widget _buildGanttView() {
    return _buildCustomGanttChart();
  }

  // Build custom Gantt chart
  Widget _buildCustomGanttChart() {
    final plannedTasks = _tasks.where((task) => task.startTime != null).toList();

    if (plannedTasks.isEmpty) {
      return const Center(child: Text('No scheduled tasks to display.'));
    }

    DateTime earliestStart = plannedTasks.first.startTime!;
    DateTime latestEnd = plannedTasks.first.startTime!.add(Duration(hours: plannedTasks.first.duration));

    for (var task in plannedTasks) {
      if (task.startTime!.isBefore(earliestStart)) {
        earliestStart = task.startTime!;
      }
      final endTime = task.startTime!.add(Duration(hours: task.duration));
      if (endTime.isAfter(latestEnd)) {
        latestEnd = endTime;
      }
    }

    earliestStart = earliestStart.subtract(const Duration(hours: 1));
    latestEnd = latestEnd.add(const Duration(hours: 1));

    final totalHours = latestEnd.difference(earliestStart).inHours;
    if (totalHours <= 0) {
      return const Center(child: Text('Error: Invalid schedule duration.'));
    }

    const hourWidth = 60.0;
    const taskHeight = 50.0;
    const labelWidth = 150.0;

    plannedTasks.sort((a, b) => a.startTime!.compareTo(b.startTime!));

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Task labels column
          SizedBox(
            width: labelWidth,
            child: Column(
              children: [
                const SizedBox(height: 50), // Align with timeline header
                ...plannedTasks.asMap().entries.map((entry) {
                  final task = entry.value;
                  return Container(
                    height: taskHeight,
                    padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
                    child: Text(
                      task.name,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                      overflow: TextOverflow.ellipsis,
                    ),
                  );
                }),
              ],
            ),
          ),
          // Gantt chart
          SizedBox(
            width: totalHours * hourWidth,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Timeline header
                SizedBox(
                  height: 50,
                  child: Row(
                    children: List.generate(
                      totalHours + 1,
                      (index) {
                        final time = earliestStart.add(Duration(hours: index));
                        final previousTime = index > 0 ? earliestStart.add(Duration(hours: index - 1)) : earliestStart;
                        return SizedBox(
                          width: hourWidth,
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                '${time.hour}:00',
                                style: const TextStyle(fontWeight: FontWeight.bold),
                              ),
                              if (index > 0 && time.day != previousTime.day)
                                Text(
                                  '${time.month}/${time.day}',
                                  style: const TextStyle(fontSize: 10),
                                ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ),
                // Gantt bars
                Container(
                  height: plannedTasks.length * taskHeight,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Stack(
                    children: [
                      // Vertical grid lines
                      ...List.generate(
                        totalHours + 1,
                        (index) => Positioned(
                          left: index * hourWidth,
                          top: 0,
                          bottom: 0,
                          child: Container(
                            width: 1,
                            color: Colors.grey.shade300,
                          ),
                        ),
                      ),
                      // Horizontal grid lines
                      ...List.generate(
                        plannedTasks.length,
                        (index) => Positioned(
                          left: 0,
                          right: 0,
                          top: (index + 1) * taskHeight,
                          child: Container(
                            height: 1,
                            color: Colors.grey.shade300,
                          ),
                        ),
                      ),
                      // Task bars
                      ...plannedTasks.asMap().entries.map((entry) {
                        final index = entry.key;
                        final task = entry.value;
                        final startOffsetHours = task.startTime!.difference(earliestStart).inHours;
                        final width = task.duration * hourWidth;

                        return Positioned(
                          left: startOffsetHours * hourWidth,
                          top: index * taskHeight + 5,
                          child: Tooltip(
                            message: '${task.name} (${task.duration}h, Priority: ${task.priority})',
                            child: GestureDetector(
                              onTap: () => _showEditTaskDialog(task),
                              child: Container(
                                width: width,
                                height: taskHeight - 10,
                                decoration: BoxDecoration(
                                  color: Colors.teal.withOpacity(0.7),
                                  borderRadius: BorderRadius.circular(4),
                                  border: Border.all(color: Colors.teal),
                                ),
                                padding: const EdgeInsets.all(4),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      task.name,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                        fontSize: 12,
                                        color: Colors.white,
                                      ),
                                    ),
                                    Text(
                                      '${task.duration}h',
                                      style: const TextStyle(fontSize: 10, color: Colors.white),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        );
                      }),
                      // Current time indicator
                      if (DateTime.now().isAfter(earliestStart) && DateTime.now().isBefore(latestEnd))
                        Positioned(
                          left: DateTime.now().difference(earliestStart).inHours * hourWidth,
                          top: 0,
                          bottom: 0,
                          child: Container(
                            width: 2,
                            color: Colors.red,
                            child: Column(
                              children: [
                                Container(
                                  width: 10,
                                  height: 10,
                                  decoration: const BoxDecoration(
                                    color: Colors.red,
                                    shape: BoxShape.circle,
                                  ),
                                  margin: const EdgeInsets.only(top: 5, left: -4),
                                ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}