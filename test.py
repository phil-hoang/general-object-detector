import sys

    
# Allow no model or selected model  
supported_models = ["-ssdm", "-ssdmlite", "-ssdvgg", "-detr", "-fasterrcnn"]
    
# Initialize log variable
logs_enabled = False

# Camera mode without a model
if (len(sys.argv) == 1):
    model_type = None
    video_file = None
    print("Camera without model")
# Camera mode with a model
#elif ((len(sys.argv) == 2 or len(sys.argv) == 3 ) and (sys.argv[1] in supported_models)):
elif (len(sys.argv) == 2 and (sys.argv[1] in supported_models)):
    model_type = sys.argv[1]
    video_file = None
    print("Camera with model")
elif (len(sys.argv) == 3 and (sys.argv[1] in supported_models) and (sys.argv[2] == "-l")):
        model_type = sys.argv[1]
        video_file = None
        logs_enabled = True
        print("Camera with model")
        print("and logging")
# Video file mode with a model
elif (len(sys.argv) == 3 and (sys.argv[1] in supported_models)):
    model_type = sys.argv[1]
    video_file = sys.argv[2]
    print("Video file with model")

elif (len(sys.argv) == 4 and (sys.argv[1] in supported_models) and (sys.argv[3] == "-l")):
        model_type = sys.argv[1]
        video_file = sys.argv[2]
        logs_enabled = True
        print("Video file with model")
        print("and logging")
else:
    print("Usage: <model> <video_filename> [opt: <-l>]\nAvailable models are: -ssdm, -ssdmlite, -ssdvgg, -fasterrcnn, -detr\nTo just run the webcam provide no args.")
    exit()

if (len(sys.argv) <= 2):
    print("\n\nStarting camera ... \nPress q to exit ")
else:
    print("\n\nStarting video ... \nPress q to exit ")