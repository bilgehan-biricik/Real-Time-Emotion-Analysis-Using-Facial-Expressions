import React, { Component } from "react";
import { AppBar, Typography } from "@material-ui/core";

class Footer extends Component {
  render() {
    return (
      <AppBar position="static" style={{ padding: "10px" }}>
        <Typography style={{ textAlign: "center" }}>
          Copyright © 2020 Bilgehan Biricik · All rights reserved
        </Typography>
      </AppBar>
    );
  }
}

export default Footer;
