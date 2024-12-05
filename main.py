import argparse
import cv2
import matplotlib.pyplot as plt
from words2contact import Words2Contact

def main(image_path, prompt, use_gpt, yello_vlm, output_path):
    # Load and process the image
    img = cv2.flip(cv2.imread(image_path), 0)

    # Display the input image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Initialize the Words2Contact model
    words2contact = Words2Contact(use_gpt=use_gpt, yello_vlm=yello_vlm)

    # Predict based on the prompt and image
    point, _, bbs, _, response = words2contact.predict(prompt, img)

    # Print prompt and response
    print("User: ", prompt)
    print("Response: ", response)

    # Visualize results
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='lower')

    for bb in bbs:
        bb.plot_bb(ax)
    ax.scatter(point.x, point.y, color='red')
    plt.savefig(output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Words2Contact with an image and a text prompt.")
    parser.add_argument("--image_path", type=str, default="data/test.png", help="Path to the input image file. Default: 'data/test.png'.")
    parser.add_argument("--prompt", type=str, default="Place your hand above the red bowl, a lot left from the banana.",
                        help="Text prompt for Words2Contact. Default: 'Place your hand above the red bowl, a lot left from the banana.'.")
    parser.add_argument("--use_gpt", action="store_true", help="use openai api for the llm, remember to export OPEANAI_KEY")
    parser.add_argument("--yello_vlm", type=str, default="GroundingDINO", help="Model to use for YELLO VLM. Default: 'GroundingDINO'.")
    parser.add_argument("--output_path", type=str, default="data/test_output.png", help="Path to save the output image. Default: 'data/test_output.png'.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args.image_path, args.prompt, args.use_gpt, args.yello_vlm, args.output_path)
