enum EndEffector{
  RIGHT_HAND="right_hand",
  LEFT_HAND="left_hand",
}

interface Correction{
    use_eef: EndEffector;
    chain_of_thought: string;
    correction_x: number;
    correction_y: number;
  }