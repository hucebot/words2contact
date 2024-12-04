from words2contact.scripts.words2contacts import Words2Contact
import cv2
import matplotlib.pyplot as plt

# test functionality
if __name__ == "__main__":
    IMAGE_PATH = "data/test.png"
    img = cv2.flip(cv2.imread(IMAGE_PATH), 0)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    words2contact = Words2Contact(use_gpt=True, yello_vlm="GroundingDINO")

    prompt = "Place your hand above the red bowl, a lot left from the banana."

    point, _, bbs, _, response= words2contact.predict(prompt, img)

    print("User: ", prompt)
    print("Response: ", response)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='lower')

    for bb in bbs:
        bb.plot_bb(ax)
    ax.scatter(point.x, point.y, color='red')
    plt.savefig("data/test_output.png")


