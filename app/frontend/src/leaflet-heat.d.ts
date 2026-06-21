declare module "leaflet.heat" {
  const _default: unknown;
  export default _default;
}

import * as L from "leaflet";
declare module "leaflet" {
  interface HeatLayerOptions {
    minOpacity?: number;
    maxZoom?: number;
    max?: number;
    radius?: number;
    blur?: number;
    gradient?: Record<number, string>;
  }
  type HeatLatLngTuple = [number, number, number?];
  interface HeatLayer extends L.Layer {
    setLatLngs(latlngs: HeatLatLngTuple[]): this;
    setOptions(options: HeatLayerOptions): this;
  }
  function heatLayer(
    latlngs: HeatLatLngTuple[],
    options?: HeatLayerOptions
  ): HeatLayer;
}
