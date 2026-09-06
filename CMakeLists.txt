cmake_minimum_required(VERSION 3.8)
project(kinova_gen3_lite_rl_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Installs the kinova_gen3_lite_rl_bridge/ Python package (observation_builder,
# safety_clamp, policy_client, ipc_protocol, policy_bridge_node) onto the ROS
# Python path so policy_bridge_node.py's absolute imports resolve at runtime.
ament_python_install_package(${PROJECT_NAME})

# Installed a second time here, renamed, as the actual ros2-run executable —
# same pattern as the sibling gen3-lite-ros2 package's per-script installs.
install(PROGRAMS
  ${PROJECT_NAME}/policy_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME policy_bridge_node
)

install(PROGRAMS
  ${PROJECT_NAME}/shared_autonomy_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME shared_autonomy_node
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# config/ also holds the uv project's tea_table_*.yaml (unrelated to this
# package) — install our files explicitly rather than the whole directory.
install(FILES
  config/safety_limits.yaml
  config/rl_bridge_params.yaml
  config/shared_autonomy.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  # This CMakeLists.txt sits at the repo root, alongside the unrelated
  # kinova_mjlab_reaching uv project (src/, scripts/, checkpoints/, ...) —
  # every linter below is scoped explicitly to this package's own files so
  # it doesn't also lint/style-check that separate project's code.
  set(_pkg_py_paths "${PROJECT_NAME}" "launch" "test")

  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(${_pkg_py_paths})

  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257(${_pkg_py_paths})

  find_package(ament_cmake_lint_cmake REQUIRED)
  ament_lint_cmake(CMakeLists.txt)

  find_package(ament_cmake_xmllint REQUIRED)
  ament_xmllint(package.xml)

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_observation_builder test/test_observation_builder.py)
  ament_add_pytest_test(test_safety_clamp test/test_safety_clamp.py)
  ament_add_pytest_test(test_ipc_roundtrip test/test_ipc_roundtrip.py)
  ament_add_pytest_test(test_goal_belief test/test_goal_belief.py)
  ament_add_pytest_test(test_joy_joint_jog test/test_joy_joint_jog.py)
  ament_add_pytest_test(test_potential_field_math test/test_potential_field_math.py)
endif()

ament_package()
