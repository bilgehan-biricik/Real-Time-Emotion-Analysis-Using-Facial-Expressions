import React, { Component } from "react";
import {
  AppBar,
  Toolbar,
  IconButton,
  Divider,
  Drawer,
  Typography,
} from "@material-ui/core";
import { Menu } from "@material-ui/icons";
import ListMenu from "./ListMenu";

class Header extends Component {
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
    );
  }
}

export default Header;
