import React from "react";
import {
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Link,
} from "@material-ui/core";
import { Dashboard, TableChart } from "@material-ui/icons";

const ListMenu = () => {
  return (
    <List style={{ width: "250px" }}>
      <Link href="/" color="inherit" underline="none">
        <ListItem button>
          <ListItemIcon>{<Dashboard />}</ListItemIcon>
          <ListItemText primary="Dashboard" />
        </ListItem>
      </Link>
      <Link href="/results" color="inherit" underline="none">
        <ListItem button>
          <ListItemIcon>{<TableChart />}</ListItemIcon>
          <ListItemText primary="Resutls" />
        </ListItem>
      </Link>
    </List>
  );
};

export default ListMenu;
