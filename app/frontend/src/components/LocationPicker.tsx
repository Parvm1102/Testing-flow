import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";

// Leaflet's default marker icons break under bundlers; build a crisp SVG pin.
const pinIcon = L.divIcon({
  className: "",
  html: `<div style="transform:translate(-50%,-100%)">
    <svg width="26" height="34" viewBox="0 0 26 34" xmlns="http://www.w3.org/2000/svg">
      <path d="M13 0C5.8 0 0 5.8 0 13c0 9.2 13 21 13 21s13-11.8 13-21C26 5.8 20.2 0 13 0z" fill="#16c79a"/>
      <circle cx="13" cy="13" r="5" fill="#0c1116"/>
    </svg></div>`,
  iconSize: [26, 34],
});

const CARTO_DARK =
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

function ClickCapture({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onPick(
        Number(e.latlng.lat.toFixed(5)),
        Number(e.latlng.lng.toFixed(5))
      );
    },
  });
  return null;
}

function Recenter({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lng], map.getZoom(), { animate: true });
  }, [lat, lng, map]);
  return null;
}

export function LocationPicker({
  lat,
  lng,
  onPick,
}: {
  lat: number;
  lng: number;
  onPick: (lat: number, lng: number) => void;
}) {
  return (
    <div className="h-56 overflow-hidden rounded-lg border border-border">
      <MapContainer
        center={[lat, lng]}
        zoom={12}
        scrollWheelZoom
        style={{ height: "100%", width: "100%" }}
        attributionControl={false}
      >
        <TileLayer
          url={CARTO_DARK}
          subdomains="abcd"
          maxZoom={19}
        />
        <Marker position={[lat, lng]} icon={pinIcon} />
        <ClickCapture onPick={onPick} />
        <Recenter lat={lat} lng={lng} />
      </MapContainer>
    </div>
  );
}
