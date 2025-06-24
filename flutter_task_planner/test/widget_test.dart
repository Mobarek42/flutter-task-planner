import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/main.dart'; // Assurez-vous que le chemin est correct

void main() {
  testWidgets('Task Planner loads and shows tabs', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const TaskPlanningApp());

    // Verify that the app title is present.
    expect(find.text('Task Planner'), findsOneWidget);

    // Verify that the "Tâches" tab is present.
    expect(find.text('Tâches'), findsOneWidget);

    // Verify that the "Planning" tab is present.
    expect(find.text('Planning'), findsOneWidget);

    // Tap the "Planifier" button (icon) and trigger a frame.
    await tester.tap(find.byIcon(Icons.schedule));
    await tester.pump();

    // Ici, on pourrait ajouter plus de vérifications après "Planifier",
    // mais pour l’instant, on vérifie juste que ça ne plante pas.
  });
}