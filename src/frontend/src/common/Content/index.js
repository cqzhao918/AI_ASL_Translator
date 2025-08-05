import React from 'react';
import { withStyles } from '@material-ui/core';
import styles from './styles';

const Content = props => {
    const classes = props.classes;
    const children = props.children;

    return (
        <div className={classes.root}>
            {children}
        </div>
    );
}

export default withStyles(styles)(Content);