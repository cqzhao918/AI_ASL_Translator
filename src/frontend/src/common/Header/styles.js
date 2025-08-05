import { fade } from '@material-ui/core/styles/colorManipulator';

const styles = theme => ({
    root: {
        flexGrow: 1,
    },
    grow: {
        flexGrow: 1,
    },
    appLink: {
        color: "inherit",
        textDecoration: "inherit",
    },
    appTitle: {
        fontSize: "1.286rem",
        lineHeight: "1.33",
        fontWeight: "800",
        letterSpacing: "3px"
    },
    menuButton: {
        marginLeft: -12,
        marginRight: 5,
    },
    verticalDivider: {
        border: "1px solid white",
        marginLeft: "5px",
        marginRight: "5px",
    },
    list: {
        width: 250,
    },
    listItemText: {
        fontSize: "15px"
    },
    search: {
        position: 'relative',
        borderRadius: theme.shape.borderRadius,
        backgroundColor: fade(theme.palette.common.white, 0.15),
        '&:hover': {
            backgroundColor: fade(theme.palette.common.white, 0.25),
        },
        marginLeft: 0,
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            marginLeft: theme.spacing(1),
            width: 'auto',
        },
    },
    searchIcon: {
        width: theme.spacing(1) * 9,
        height: '100%',
        position: 'absolute',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    inputRoot: {
        color: 'inherit',
        width: '100%',
    },
    inputInput: {
        paddingTop: theme.spacing(1),
        paddingRight: theme.spacing(1),
        paddingBottom: theme.spacing(1),
        paddingLeft: theme.spacing(1) * 10,
        transition: theme.transitions.create('width'),
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            width: '35ch',
            '&:focus': {
                width: '70ch',
            },
        },
    },
    alertContainer: {
        position: "fixed",
        top: "0",
        left: "40%",
        zIndex: 1000,
    },
});


export default styles;