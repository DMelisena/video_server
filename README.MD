TODO: 
<!-- make download protocol -->
make sure that the link that is given after the POST is done downloadable
tryout what's inside the zip using the mock app that I made
make see all package http request
make delete http request
make deleteAll http request
make a limit on fileSize on the App

resort :
only extract one image on each first dbscan result
reapply dbscan on the images

example command to build docker 
```
docker build -t dmelisena/videodocker:0.1 .
```
example on running the docker :

```
ngrok http --domain=prime-whole-fish.ngrok-free.app 8080

```
ngrok would have this as the forwarding
https://prime-whole-fish.ngrok-free.app -> http://localhost:8080

to turn on the server on localhost:8080, run
```
docker run -p 8080:8080 0cef3bbb6aae

```
last is IMAGE ID inside which can be checked using this command
```
docker images
```
