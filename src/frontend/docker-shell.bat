SET IMAGE_NAME=capy-app-frontend-react
SET BASE_DIR=%cd%

docker build -t %IMAGE_NAME% -f Dockerfile.dev .
docker run  --rm --name %IMAGE_NAME% -ti --mount type=bind,source="%cd%",target=/app -p 3000:3000 %IMAGE_NAME%