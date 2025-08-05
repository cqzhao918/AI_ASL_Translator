import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';

import DataService from "../../services/DataService";
import styles from './styles';

const Home = (props) => {
    const { classes } = props;

    console.log("================================== Home ======================================");

    const inputFile = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);

    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleImageUploadClick = () => {
        inputFile.current.click();
    }
    // const handleOnChange = (event) => {
    //     console.log(event.target.files);
    //     setImage(URL.createObjectURL(event.target.files[0]));

    //     var formData = new FormData();
    //     formData.append("file", event.target.files[0]);
    //     DataService.Predict(formData)
    //         .then(function (response) {
    //             console.log(response.data);
    //             setPrediction(response.data);
    //         })
    // }
    const handleOnChange = (event) => {
        const file = event.target.files[0];
        const url = URL.createObjectURL(file);
        setImage(url);

        // Link to API
        var formData = new FormData();
        formData.append("file", event.target.files[0]);
        DataService.Predict(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
        })
      };
    

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.container}>
                    {prediction &&
                        <Typography variant="h4" gutterBottom align='center'>
                            
                            <span className={classes.safe}>{prediction.prediction_label}</span>
                            
                        </Typography>
                    }
                    {/* <div className={classes.dropzone} onClick={() => handleImageUploadClick()}> */}
                    <div className={classes.dropzone}>

                        <input
                            type="file"
                            accept="video/*"
                            capture="camera"
                            on
                            autocomplete="off"
                            tabindex="-1"
                            className={classes.fileInput}
                            ref={inputFile}
                            onChange={(event) => handleOnChange(event)}
                            accept=".mov,.mp4"
                        />
                        {!image && <button onClick={handleImageUploadClick}>Choose</button>}
                        {image && (
                            <video
                            className={classes.preview}
                            src = {image}
                            width="100%"
                            controls
                          />

                        )}
                        {/* <div><video className={classes.preview} src={image} /></div> */}
                        <div className={classes.help}>Click to upload a video...</div>
                    </div>
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Home);