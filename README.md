# MonitoraUFF - Computer Vision Plugins

This repository includes sophisticated Computer Vision plugins developed for the MonitoraUFF system.

## Overview

The plugins implemented in this repository leverages advanced computer vision techniques to accurately detect and recognize license plates, persons, animals, objets and actions in various conditions. They are essential components of the MonitoraUFF system, providing real-time analysis and enhancing the platform's monitoring capabilities.

## Getting Started

To get started with any plugin, follow these steps:

1. Make sure you have [Docker](https://www.docker.com/) installed.

2. Make sure your system has a [CUDA-capable GPU](https://docs.nvidia.com/gameworks/content/developertools/desktop/view_device_information.htm).

3. Build the Docker image:

    ```bash
    docker build -t cv_plugins https://github.com/MonitoraUFF/MonitoraUFF-ComputerVisionPlugins.git
    ```

4. Use the commands presented in the following subsections to start a plugin. Specify the target GPUs by setting the flag `--gpus` (see [docker.docs](https://docs.docker.com/config/containers/resource_constraints/#gpu) for details). Use the tags `--server` and `--video` to address, respectively, the MonitoraUFF server and the camera. Set the `--name` flag to identify the plugin's agent.

### Licence Plate Recognition

Run the License Plate Agent on the Docker image:

```bash
docker run -it --rm -v ${pwd}/plugins-storage:/app/plugins-storage --gpus all cv_plugins licenceplate --server SERVER_ADDRESS --video VIDEO_URI --name TAG_SLUG
```
