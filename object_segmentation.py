import argparse
import cv2
import matplotlib.pyplot as plt
from words2contact import Saygment

def main(image_path, prompt, saygment_vlm, output_path):
    # Load and process the image
    img = cv2.flip(cv2.imread(image_path), 0)

    # Initialize the Words2Contact model
    saygment = Saygment(saygment_vlm)

    # Predict based on the prompt and image
    point, heatmap = saygment.predict(img, [prompt])

    # Print prompt and response
    print("User: ", prompt)


    # Visualize results
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='lower')
    seg = ax.imshow(heatmap, alpha=0.5, cmap='inferno', origin='lower')


    # Remove axis
    ax.axis('off')
    plt.tight_layout()

    # set title and super title
    ax.set_title(f"Prompt: \"{prompt}\"", fontsize=10)
    fig.suptitle(f"Saygment VLM: {saygment_vlm}", fontsize=16)
    plt.colorbar(seg)
    plt.savefig(output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Words2Contact with an image and a text prompt.")
    parser.add_argument("--image_path", type=str, default="data/test.png", help="Path to the input image file. Default: 'data/test.png'.")
    parser.add_argument("--prompt", type=str, default="bowl", help="Text prompt of the object to be segmented'.")
    parser.add_argument("--saygment_vlm", type=str, default="CLIPSeg", help="Model to use for Saygment VLM. Default: 'GroundingDINO'.")
    parser.add_argument("--output_path", type=str, default="data/test_output.png", help="Path to save the output image. Default: 'data/test_output.png'.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args.image_path, args.prompt, args.saygment_vlm, args.output_path)
