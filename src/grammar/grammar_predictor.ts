interface Response {
    chain_of_thought: ChainOfThought;
}

interface ChainOfThought {
    analysis: string;
    task: Task
}

interface Task {
    target_x: number
    target_y: number
}

