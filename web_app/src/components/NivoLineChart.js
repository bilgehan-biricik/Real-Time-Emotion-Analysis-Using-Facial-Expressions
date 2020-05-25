import React from "react";
import { ResponsiveLine } from "@nivo/line";

const NivoLineChart = (props) => {
  return (
    <div style={{ width: "960px", height: "500px" }}>
      <ResponsiveLine
        data={props.data}
        margin={{ top: 50, right: 100, bottom: 60, left: 75 }}
        yScale={{
          type: "linear",
        }}
        xScale={{
          type: "linear",
          min: "auto",
        }}
        axisBottom={{
          orient: "bottom",
          tickSize: 8,
          tickPadding: 5,
          tickRotation: 0,
          tickValues: 10,
          legend: "Time (sec)",
          legendOffset: 48,
          legendPosition: "middle",
        }}
        axisLeft={{
          orient: "left",
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: "Satisfaction (%)",
          legendOffset: -60,
          legendPosition: "middle",
        }}
        colors={{ scheme: "paired" }}
        theme={{
          fontSize: "16px",
          axis: {
            legend: {
              text: {
                fontSize: "16px",
              },
            },
          },
        }}
        enablePoints={false}
        enableGridX={true}
        curve="monotoneX"
        animate={false}
        isInteractive={true}
        enableSlices="x"
        useMesh={true}
        legends={[
          {
            anchor: "bottom-right",
            direction: "column",
            justify: false,
            translateX: 100,
            translateY: 0,
            itemsSpacing: 0,
            itemDirection: "left-to-right",
            itemWidth: 80,
            itemHeight: 20,
            itemOpacity: 0.75,
            symbolSize: 12,
            symbolShape: "circle",
            symbolBorderColor: "rgba(0, 0, 0, .5)",
            effects: [
              {
                on: "hover",
                style: {
                  itemBackground: "rgba(0, 0, 0, .03)",
                  itemOpacity: 1,
                },
              },
            ],
          },
        ]}
      />
    </div>
  );
};

export default NivoLineChart;
