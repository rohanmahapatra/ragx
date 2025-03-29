### Tips for Docker Usage

Here are some useful commands and tips to help you manage Docker containers during your workflow.

1. **Building the Docker Image**:
   After cloning the repository and navigating to the artifact folder, you can build the Docker image using the following command:
   ```bash
   ./build_docker.sh
   ```
   This command will build the image and tag it as `artifact-app:latest`.

2. **Running the Docker Container in Interactive Mode**:
   To run the container interactively (so you can interact with it), use this command:
   ```bash
   docker run -it artifact-app:latest /bin/bash
   ```
   This will start the container and give you access to a Bash shell within the container.

3. **Exiting the Docker Container**:
   Once you're done working inside the container, you can exit the interactive session by typing:
   ```bash
   exit
   ```
   This will leave the container running in the background, but you will be returned to your host system’s shell.

4. **Re-entering a Running Docker Container**:
   If the container is still running and you want to re-enter it, follow these steps:
   - First, list the running containers:
     ```bash
     docker ps
     ```
   - Then, attach to the container using its ID or name:
     ```bash
     docker exec -it <container_id_or_name> /bin/bash
     ```
     This will start a new interactive session inside the container.

5. **Stopping the Docker Container**:
   When you no longer need the container to be running, you can stop it by running:
   ```bash
   docker stop <container_id_or_name>
   ```
   This will gracefully stop the container.

6. **Removing the Stopped Container**:
   If you want to remove the container after stopping it (to free up resources), run:
   ```bash
   docker rm <container_id_or_name>
   ```
   This will remove the container from your system. Make sure the container is stopped before removing it.

7. **Killing a Docker Container**:
   If the container is not responding or you need to forcefully stop it, you can "kill" the container:
   ```bash
   docker kill <container_id_or_name>
   ```
   This will immediately stop the container without waiting for it to shut down gracefully.

8. **Viewing Docker Logs**:
   If you want to check the logs of a container to troubleshoot or see what's happening inside, you can use the following command:
   ```bash
   docker logs <container_id_or_name>
   ```
