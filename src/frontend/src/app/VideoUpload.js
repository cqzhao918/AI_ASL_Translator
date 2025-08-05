import React from "react";
import "./styles.css";

import Card from "@material-ui/core/Card";
import Grid from "@material-ui/core/Grid";
import Input from "@material-ui/core/Input";
import Typography from "@material-ui/core/Typography";

import LocalForage from "localforage/dist/localforage.js";
const videoStore = LocalForage.createInstance({ name: "VideoStore" });

export default function App() {
  setVideo();

  return (
    <Card className="App">
      <Grid container spacing={2} direction="column">
        <Grid item>
          <Typography variant="h4">Video Upload</Typography>
        </Grid>
        <Grid item>
          <Typography variant="h6">
            Save the video to indexDB, load from indexDB, then play.
          </Typography>
        </Grid>
        <Grid item>
          <Input
            id="input"
            name="input"
            type="file"
            accept="video/*"
            onChange={() => {
              let file = document.getElementById("input").files[0];
              if (file instanceof File) {
                file = new Blob([file], { type: file.type });
                videoStore
                  .setItem("video", file)
                  .then(() => {
                    setVideo();
                  })
                  .catch(err => console.error("Unable to store video", err));
              }
            }}
          />
        </Grid>
        <Grid item>
          <div id="video" />
        </Grid>
      </Grid>
    </Card>
  );
}

function setVideo() {
  videoStore
    .getItem("video")
    .then(val => {
      if (val) {
        let vid = document.createElement("video");
        vid.src = URL.createObjectURL(val);
        vid.muted = true;
        vid.style = { maxWidth: "400px", maxHeight: "400px" };
        vid.autoPlay = true;
        vid.controls = true;
        vid.playsInline = true;

        // creating and adding the element appears to be
        // the issue... When just setting the source of
        // an element it seems to work for a while. But
        // if left alone, the video will eventually stop
        // playing or allowing time scrubbing/seeking.
        let elem = document.getElementById("video");
        while (elem.children.length > 0) {
          if (elem.firstChild.src) {
            URL.revokeObjectURL(elem.firstChild.src);
          }
          elem.removeChild(elem.firstChild);
        }
        elem.appendChild(vid);
      }
    })
    .catch(err => {
      console.error("Unable to retrieve video", err);
    });
}
