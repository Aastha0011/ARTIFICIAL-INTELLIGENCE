{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOHMeUWDb8JTEhwvTVvjCUu",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Aastha0011/CAT-and-DOG-model/blob/main/train.model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AyG0rBEf7EGK",
        "outputId": "f6c38b28-fa04-4872-a84e-3afcb02f7826"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Images 51\n",
            "Model Saved Successfully!\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import cv2\n",
        "import joblib\n",
        "import numpy as np\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "\n",
        "# Dataset path\n",
        "dataset_path = r\"/content/\"\n",
        "\n",
        "images = []\n",
        "labels = []\n",
        "\n",
        "classes = [\"CAT\", \"DOG\"]\n",
        "\n",
        "IMG_SIZE = 64\n",
        "\n",
        "for label, folder in enumerate(classes):\n",
        "\n",
        "    folder_path = os.path.join(dataset_path, folder)\n",
        "\n",
        "    for file in os.listdir(folder_path):\n",
        "\n",
        "        img_path = os.path.join(folder_path, file)\n",
        "\n",
        "        img = cv2.imread(img_path)\n",
        "\n",
        "        if img is None:\n",
        "            continue\n",
        "\n",
        "        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))\n",
        "\n",
        "        img = img.flatten()\n",
        "\n",
        "        images.append(img)\n",
        "\n",
        "        labels.append(label)\n",
        "\n",
        "\n",
        "X = np.array(images)\n",
        "y = np.array(labels)\n",
        "\n",
        "print(\"Training Images\", len(X))\n",
        "\n",
        "model = LogisticRegression(max_iter=1000)\n",
        "\n",
        "model.fit(X, y)\n",
        "\n",
        "joblib.dump(model, \"cat_dog_model.pkl\")\n",
        "\n",
        "print(\"Model Saved Successfully!\")"
      ]
    }
  ]
}