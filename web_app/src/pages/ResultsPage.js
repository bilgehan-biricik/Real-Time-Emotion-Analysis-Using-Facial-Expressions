import React, { Component } from "react";
import {
  Container,
  Paper,
  TableContainer,
  Table,
  TableRow,
  TableCell,
  TableHead,
  TableBody,
} from "@material-ui/core";
import axios from "axios";
import Header from "../components/Header";
import Footer from "../components/Footer";

export default class ResultsPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
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
        <Header />
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
        <Footer />
      </div>
    );
  }
}
