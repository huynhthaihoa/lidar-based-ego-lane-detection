import yaml

from glob import glob
from lane_detector import LaneDetector

# Output reproduction

if __name__ == "__main__":
    
    # Read configuration from config file
    args = yaml.safe_load(open("configs.yml"))

    # LaneDetector initialization & configuration
    detector = LaneDetector(figure_dir = args["figure_dir"], 
                           output_dir = args["output_dir"],
                           distance_threshold=args["distance_threshold"],
                           num_iterations=args["num_iterations"],
                           intensity_bin_res=args["intensity_bin_res"],
                           horizontal_bin_res=args["horizontal_bin_res"],
                           lane_width=args["lane_width"],
                           lane_bound_threshold=args["lane_bound_threshold"],
                           degree=args["degree"],
                           num_lanes=args["num_lanes"])
    
    input_paths = glob("pointclouds/*.bin")

    for input_path in input_paths:
        print(f"Processing file {input_path}...\n")
        
        # Run the end-to-end lane detection pipeline on each input bin file
        detector.pipeline(input_path)
        
        print(f"Finish processing file {input_path}!\n")
        print("--------------------")