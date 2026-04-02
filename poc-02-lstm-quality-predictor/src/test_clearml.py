from clearml import Task

# This line starts the automation engine
task = Task.init(project_name='Maestro_Test', task_name='Connection_Check')

print("Sending a test metric to the cloud...")
task.get_logger().report_scalar("Check", "Status", iteration=1, value=100)

task.close()
print("Success! Check your dashboard.")