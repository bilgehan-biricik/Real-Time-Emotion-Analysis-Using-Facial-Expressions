import React, { Component } from "react";
import {
  AppBar,
  Toolbar,
  IconButton,
  Divider,
  Typography,
  Container,
  Paper,
  Drawer,
  TableContainer,
  Table,
  TableRow,
  TableCell,
  TableHead,
  TableBody,
} from "@material-ui/core";
import ListMenu from "../components/ListMenu";
import { Menu } from "@material-ui/icons";
import axios from "axios";

export default class ResultsPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isDrawerOpen: false,
      rows: [],
    };
  }

  componentDidMount() {
    let rows = [];
    axios
      .get("http://127.0.0.1:5000/api/get-session-results")
      .then((response) => {
        response.data.results.forEach((el) => {
          rows.push(
            this.createData(el[0], el[1], el[2], el[3], el[4], el[5], el[6])
          );
        });
        this.setState({ rows: rows });
      })
      .catch((err) => console.log(err));
  }

  toggleDrawer = (open) => (event) => {
    if (
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }
    this.setState({ isDrawerOpen: open });
  };

  createData = (
    id,
    sessionStartTime,
    duration,
    maxDetectedFace,
    avgSatisfaction,
    mostDetectedEmotion,
    mostDetectedEmotionCounter
  ) => {
    return {
      id,
      sessionStartTime,
      duration,
      maxDetectedFace,
      avgSatisfaction,
      mostDetectedEmotion,
      mostDetectedEmotionCounter,
    };
  };

  render() {
    return (
      <div style={{ backgroundColor: "#F6F6F6" }}>
        <AppBar position="static">
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={this.toggleDrawer(true)}
            >
              <Menu />
            </IconButton>
            <Divider />
            <Drawer
              anchor="left"
              open={this.state.isDrawerOpen}
              onClose={this.toggleDrawer(false)}
            >
              <ListMenu />
            </Drawer>
            <Typography variant="h5">Facial Expression Recognition</Typography>
          </Toolbar>
        </AppBar>
        <Container
          maxWidth="xl"
          style={{ marginTop: "20px", marginBottom: "20px" }}
        >
          <Paper
            elevation={0}
            style={{ padding: "20px", backgroundColor: "#F6F6F6" }}
          >
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell align="center">Session Start</TableCell>
                    <TableCell align="center">Duration (sec)</TableCell>
                    <TableCell align="center">Max. Detected Face</TableCell>
                    <TableCell align="center">
                      Most Detected Emotion
                    </TableCell>
                    <TableCell align="center">
                    Number of Repetitions
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {this.state.rows.map((row) => (
                    <TableRow key={row.id}>
                      <TableCell component="th" scope="row">
                        {row.id}
                      </TableCell>
                      <TableCell align="center">
                        {row.sessionStartTime}
                      </TableCell>
                      <TableCell align="center">{row.duration}</TableCell>
                      <TableCell align="center">{row.maxDetectedFace}</TableCell>
                      <TableCell align="center">
                        {row.mostDetectedEmotion}
                      </TableCell>
                      <TableCell align="center">
                        {row.mostDetectedEmotionCounter}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Container>
        <AppBar position="static" style={{ padding: "10px" }}>
          <Typography style={{ textAlign: "center" }}>
            Copyright © 2020 Bilgehan Biricik · All rights reserved
          </Typography>
        </AppBar>
      </div>
    );
  }
}
