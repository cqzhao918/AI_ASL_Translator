import React, { useEffect, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableRow from '@material-ui/core/TableRow';
import NumberFormat from "react-number-format";

import DataService from "../../services/DataService";
import styles from './styles';

const Currentmodel = (props) => {
    const { classes } = props;

    console.log("================================== Currentmodel ======================================");

    // Component States
    const [model, setModel] = useState(null);
    const loadModel = () => {
        DataService.GetCurrentmodel()
            .then(function (response) {
                console.log(response.data);
                setModel(response.data.model_details);
            });
    }

    // Setup Component
    useEffect(() => {
        loadModel();
    }, []);

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="sm" className={classes.container}>
                    <Typography variant="h5" gutterBottom>
                        Transformer-based Model Details:
                    </Typography>
                    <Divider />
                    {model &&
                        <TableContainer component={Paper}>
                            <Table>
                                <TableBody>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Name</Typography></TableCell>
                                        <TableCell>{model.model_name}</TableCell>
                                    </TableRow>

                                    <TableRow>
                                        <TableCell><Typography variant="h6">Trainable Parameters</Typography></TableCell>
                                        <TableCell>
                                            <NumberFormat
                                                value={model.trainable_parameters}
                                                displayType="text"
                                                thousandSeparator={true}
                                            />
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Training Time (mins)</Typography></TableCell>
                                        <TableCell>
                                            <NumberFormat
                                                value={model.execution_time}
                                                displayType="text"
                                                decimalSeparator="."
                                                decimalScale={2}
                                            />
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Loss</Typography></TableCell>
                                        <TableCell>
                                            <NumberFormat
                                                value={model.loss}
                                                displayType="text"
                                                format="####"
                                            />
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Accuracy</Typography></TableCell>
                                        <TableCell>
                                            <NumberFormat
                                                value={model.accuracy * 100.00}
                                                displayType="text"
                                                decimalSeparator="."
                                                decimalScale={2}
                                                suffix="%"

                                            />
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Model Size (Mb)</Typography></TableCell>
                                        <TableCell>
                                            <NumberFormat
                                                value={model.model_size / 1000000.00}
                                                displayType="text"
                                                decimalSeparator="."
                                                decimalScale={2}

                                            />
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Learning Rate</Typography></TableCell>
                                        <TableCell>
                                            {model.learning_rate}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Batch Size</Typography></TableCell>
                                        <TableCell>
                                            {model.batch_size}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Epochs</Typography></TableCell>
                                        <TableCell>
                                            {model.epochs}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="h6">Optimizer</Typography></TableCell>
                                        <TableCell>
                                            {model.optimizer}
                                        </TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </TableContainer>
                    }
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Currentmodel);