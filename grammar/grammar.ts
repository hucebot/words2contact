enum EndEffector{
  RIGHT_HAND="right_hand",
  LEFT_HAND="left_hand",
}
enum TaskType {
    CONACT_POINT = "contact",
    REACH = "reach"
}

interface resposne {
  chain_of_thought: string;
  tasks: Tasks[];
}

interface Tasks{
  use_eef: EndEffector;
  task_type: TaskType;
  target_x: number;
  target_y: number;
}

