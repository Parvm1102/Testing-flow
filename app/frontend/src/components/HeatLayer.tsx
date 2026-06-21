import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet.heat";

interface HeatLayerProps {
  points: L.HeatLatLngTuple[];
  radius?: number;
  blur?: number;
  max?: number;
  gradient?: Record<number, string>;
}

/** Imperative wrapper around leaflet.heat (not part of react-leaflet). */
export function HeatLayer({
  points,
  radius = 22,
  blur = 18,
  max = 1,
  gradient,
}: HeatLayerProps) {
  const map = useMap();

  useEffect(() => {
    const layer = L.heatLayer(points, {
      radius,
      blur,
      max,
      minOpacity: 0.25,
      maxZoom: 17,
      gradient,
    });
    layer.addTo(map);
    return () => {
      layer.remove();
    };
  }, [map, points, radius, blur, max, gradient]);

  return null;
}
