<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<!-- <br /> -->
<div align="center">
  <a href="https://github.com/Aeidle/EAR-Fatigue-Detection">
    <img src="assets/banner.png" alt="Logo" width="1080">
  </a>

  <h3 align="center">EAR Fatigue Detection</h3>

  <p align="center">
    Implementation of a system to detect fatigue using EAR (Eye Aspect Ratio).
    <br />
    <a href="https://github.com/Aeidle/EAR-Fatigue-Detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://drive.google.com/file/d/1JTUFBA4V8-nqCR1i2cf1AUbz2_LWZbqc/view">View Demo</a>
    ·
    <a href="https://github.com/Aeidle/EAR-Fatigue-Detection/issues">Report Bug</a>
    ·
    <a href="https://github.com/Aeidle/EAR-Fatigue-Detection/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Fatigue is a common problem among students, especially in long and intensive learning sessions. It can affect their performance, concentration, and well-being. Therefore, it is important to monitor and prevent fatigue in the classroom. In this project, we propose a system that can detect fatigue among students using cameras and a metric called eye aspect ratio (EAR). EAR is a measure of eye openness, calculated from the positions of six facial landmarks around the eye. A low EAR indicates that the eye is closed, while a high EAR indicates that the eye is open. By tracking the EAR values over time, we can determine the duration of eye closure, which is an indicator of fatigue. We aim to set a minimum threshold for EAR and alert the teacher or the supervisor when the fatigue is detected for a prolonged period. This system can help to improve the safety and the quality of learning in the classroom.

[![EAR Fatigue Detection][detected-screenshot]](assets/detected.png)

To implement this system, we will follow the steps outlined in the web page context. First, we will extract the images from a video of a learning session. Then, we will convert the images to gray scale and apply some noise removal and quality enhancement techniques. Next, we will use edge detection methods such as Canny or Sobel to find the contours of the eyes. After that, we will apply some morphological operations to refine the contours and segment the eyes. Then, we will use a facial landmark detector to locate the six points of interest (p1, …, p6) around each eye. The EAR is calculated from the formula:

$$
    EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 ||p1 - p4||}
$$

where ∣∣⋅∣∣ denotes the Euclidean distance between two points. At each time instant t (e.g., t = 10s), we will compute the EAR and save it in a CSV file3. Finally, we will analyze the EAR data to detect the periods of prolonged fatigue. We will define some fatigue thresholds based on the EAR values and set up an alert mechanism to signal the moments when the fatigue is detected for a long time. This alert can be a visual or auditory notification for the teacher or the supervisor, so that they can take appropriate actions to ensure the safety and well-being of the students.

[![EAR Fatigue Detection][detected-face]](assets/detected.png)

### Video Demonstration

https://youtu.be/qOPrmq3pk9E

### Result Chart
[![Result Chart][result-chart]](assets/plot.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With


* [![Python][Python.py]][Python-url]
* [![Dlib][Dlib.com]][Dlib-url]
* [![OpenCV][OpenCV.com]][OpenCV-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You need to have ffmpeg installed on your system, you can download it from the official website: https://www.ffmpeg.org/download.html

* For linux:
```bash
apt-get -y update && apt-get -y install ffmpeg
pip3 install --upgrade pip setuptools
pip3 install ffmpeg-python
```

* for Windows
* Install FFMPEG from [here](https://www.ffmpeg.org/download.html). Make sure to add it to system path. You can refer to this awesome tutorial for downloading and installing ffmpeg on windows [youtube-tutorial](https://www.youtube.com/watch?v=jZLqNocSQDM).

### Installation

**1. Clone the repo**
```bash
git clone https://github.com/Aeidle/EAR-Fatigue-Detection.git
```
**2. Create a virtual environnement.**

```bash
# In windows
python -m venv venv
venv\Scripts\activate
```

```bash
# In linux
python -m venv venv
source venv/bin/activate
```

**Or even better use anaconda**

```bash
conda create --name venv python=3.9
conda activate venv
```

**3. Install the dependencies using pip :**
```bash 
pip  install dlib_installation/dlib-19.22.99-cp39-cp39-win_amd64.whl
pip install -r requirements.txt
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

To run the application, you can simply execute the main file:

* **using jupyter notebook (from video file)**

just open `main.ipynb` and execute all the cells.
make sure to use the correct path for your video `cap = cv2.VideoCapture("videos/video3.MOV")`.

* **using python and CLI (from live camera)**
```bash
python app.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Adil Alami - [@MeAeidle](https://twitter.com/MeAeidle) - aeidle.me@gmail.com

Project Link: [https://github.com/Aeidle/Fatigue-Detection](https://github.com/Aeidle/EAR-Fatigue-Detection.git)

LinkedIn: [https://www.linkedin.com/in/adil-alami/](https://www.linkedin.com/in/adil-alami)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

















[detected-screenshot]: assets/detected.png
[detected-face]: assets/face-detected.png
[detected-video]: output/output_video.mp4
[result-chart]: assets/plot.png

[Python.py]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Dlib.com]: https://img.shields.io/badge/dlib-lightorange?style=for-the-badge&logo=dlib
[Dlib-url]: http://dlib.net/
[OpenCV.com]: https://img.shields.io/badge/OpenCV-orange?style=for-the-badge&logo=opencv
[OpenCV-url]: https://opencv.org/

[contributors-shield]: https://img.shields.io/github/contributors/Aeidle/EAR-Fatigue-Detection?style=for-the-badge
[contributors-url]: https://github.com/Aeidle/EAR-Fatigue-Detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Aeidle/EAR-Fatigue-Detection?style=for-the-badge
[forks-url]: https://github.com/Aeidle/EAR-Fatigue-Detection/network/members
[stars-shield]: https://img.shields.io/github/stars/Aeidle/EAR-Fatigue-Detection?style=for-the-badge
[stars-url]: https://github.com/Aeidle/EAR-Fatigue-Detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/Aeidle/EAR-Fatigue-Detection?style=for-the-badge
[issues-url]: https://github.com/Aeidle/EAR-Fatigue-Detection/issues
[license-shield]: https://img.shields.io/github/license/Aeidle/EAR-Fatigue-Detection?style=for-the-badge
[license-url]: https://github.com/Aeidle/EAR-Fatigue-Detection/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/adil-alami