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
} from "@material-ui/core";
import ListMenu from "../components/ListMenu";
import { Menu } from "@material-ui/icons";

export default class ResultsPage extends Component {
  constructor(props) {
    super(props);
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
              asdasd
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
