import React from "react";
import { Switch, Route, BrowserRouter } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import ResultsPage from "./pages/ResultsPage";

const AppRouter = () => {
  return (
    <BrowserRouter>
      <Switch>
        <Route path="/" exact component={DashboardPage} />
        <Route path="/results" exact component={ResultsPage} />
      </Switch>
    </BrowserRouter>
  );
};

export default AppRouter;
