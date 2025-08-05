REM Define some environment variables
SET IMAGE_NAME="capy-app-api-service"

REM Build the image based on the Dockerfile
docker build -t %IMAGE_NAME% -f Dockerfile .

REM Run the container
cd ..
docker run  --rm --name %IMAGE_NAME% -ti ^
            --mount type=bind,source="%cd%\api-service",target=/app ^
            --mount type=bind,source="%cd%\..\..\local_model",target=/local_model ^
            --mount type=bind,source="%cd%\..\..\secrets",target=/secrets ^
            -p 9000:9000 -e DEV=1 %IMAGE_NAME%