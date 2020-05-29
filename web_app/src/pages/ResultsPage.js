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

export default class ResultsPage extends Component {
  constructor(props) {
    super(props);
    this.rows = [
      this.createData("Frozen yoghurt", 159, 6.0, 24, 4.0),
      this.createData("Ice cream sandwich", 237, 9.0, 37, 4.3),
      this.createData("Eclair", 262, 16.0, 24, 6.0),
      this.createData("Cupcake", 305, 3.7, 67, 4.3),
      this.createData("Gingerbread", 356, 16.0, 49, 3.9),
    ];
    this.state = {
      isDrawerOpen: false,
    };
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

  createData = (name, calories, fat, carbs, protein) => {
    return { name, calories, fat, carbs, protein };
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
                    <TableCell>Dessert (100g serving)</TableCell>
                    <TableCell align="right">Calories</TableCell>
                    <TableCell align="right">Fat&nbsp;(g)</TableCell>
                    <TableCell align="right">Carbs&nbsp;(g)</TableCell>
                    <TableCell align="right">Protein&nbsp;(g)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {this.rows.map((row) => (
                    <TableRow key={row.name}>
                      <TableCell component="th" scope="row">
                        {row.name}
                      </TableCell>
                      <TableCell align="right">{row.calories}</TableCell>
                      <TableCell align="right">{row.fat}</TableCell>
                      <TableCell align="right">{row.carbs}</TableCell>
                      <TableCell align="right">{row.protein}</TableCell>
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
