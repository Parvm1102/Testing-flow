import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "leaflet/dist/leaflet.css";
import "./index.css";
import { ThemeProvider } from "./components/ThemeProvider";
import { Layout } from "./components/Layout";
import Predict from "./pages/Predict";
import MapView from "./pages/MapView";
import TopAreas from "./pages/TopAreas";
import Models from "./pages/Models";

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Layout>
        <Predict />
      </Layout>
    ),
  },
  {
    path: "/map",
    element: (
      <Layout>
        <MapView />
      </Layout>
    ),
  },
  {
    path: "/areas",
    element: (
      <Layout>
        <TopAreas />
      </Layout>
    ),
  },
  {
    path: "/models",
    element: (
      <Layout>
        <Models />
      </Layout>
    ),
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </React.StrictMode>
);
