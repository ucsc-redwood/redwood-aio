import re

def parse_input(input_str):
    """
    Parse an input string with the following possible format:
    
      {Backend}_{Application}/{StageInfo}[/{NumThreads}]
      
    Where:
      - Backend is one of {OMP, CUDA, VK}
      - Application is one of {CifarDense, CifarSparse, Tree}
      - StageInfo is either:
          "Baseline" optionally followed by a tail (e.g., "_median", "_cv", "_stddev")
          OR
          "StageX" optionally with an underscore and a core type (e.g., "Stage4_big")
          In the case of "Baseline", stage is 0.
      - NumThreads is an integer, optionally followed by tailing data (which are ignored).
      
    Examples:
      - "OMP_CifarDense/Baseline/2"               -> backend: "OMP",  application: "CifarDense", stage: 0, core_type: None, num_threads: 2
      - "OMP_CifarSparse/Baseline/1_median"         -> backend: "OMP",  application: "CifarSparse", stage: 0, core_type: None, num_threads: 1
      - "OMP_Tree/Stage2_big/1_cv"                  -> backend: "OMP",  application: "Tree", stage: 2, core_type: "big", num_threads: 1
      - "VK_Tree/Stage6_cv"                         -> backend: "VK",   application: "Tree", stage: 6, core_type: None, num_threads: None

    Returns:
        dict: A dictionary with keys:
            - 'backend'
            - 'application'
            - 'stage'
            - 'core_type'
            - 'num_threads'
    """
    # Split by '/'
    segments = input_str.split('/')
    if len(segments) < 2:
        raise ValueError("Input must have at least two segments separated by '/'")

    # Parse the first segment: "Backend_Application"
    try:
        backend, application = segments[0].split('_', 1)
    except ValueError:
        raise ValueError("First segment must be in the format 'Backend_Application'")

    # Initialize defaults
    stage = None
    core_type = None
    num_threads = None

    # Parse the stage segment (second segment)
    stage_segment = segments[1]
    if stage_segment.startswith("Baseline"):
        stage = 0
    elif stage_segment.startswith("Stage"):
        # Remove "Stage" and capture the stage number and optional core type.
        # This regex captures one or more digits and an optional underscore followed by word characters.
        m = re.match(r"Stage(\d+)(?:_(\w+))?", stage_segment)
        if m:
            stage = int(m.group(1))
            core_candidate = m.group(2)
            # Only assign core_type if it is one of the allowed types.
            if core_candidate in {"little", "small", "big"}:
                core_type = core_candidate
        else:
            raise ValueError("Stage segment does not match expected format (e.g., 'Stage4_big')")
    else:
        raise ValueError("Stage segment must start with 'Baseline' or 'Stage'")

    # Parse the number of threads if the third segment exists.
    if len(segments) > 2:
        thread_segment = segments[2]
        # Extract the leading number ignoring any tailing data (like _median, _cv, _stddev, etc.)
        m = re.match(r"(\d+)", thread_segment)
        if m:
            num_threads = int(m.group(1))
    
    return {
        "backend": backend,
        "application": application,
        "stage": stage,
        "core_type": core_type,
        "num_threads": num_threads,
    }

# Example usage:
if __name__ == "__main__":
    test_inputs = [
        "OMP_CifarDense/Baseline/2",
        "OMP_CifarSparse/Baseline/1_median",
        "OMP_CifarSparse/Baseline/4_median",
        "OMP_Tree/Stage2_big/1_cv",
        "OMP_CifarDense/Stage4_big/2_stddev",
        "VK_CifarDense/Baseline",
        "VK_CifarDense/Baseline_mean",
        "VK_Tree/Stage6_cv"
    ]
    
    for inp in test_inputs:
        result = parse_input(inp)
        print(f"Input: {inp}\nParsed: {result}\n")
