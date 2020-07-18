import React from "react";
import ReactWebcam from "react-webcam";
import ReactPlayer from "react-player";
import captureVideoFrame from "capture-video-frame";
import axios from "axios";

import NivoPieChart from "../components/NivoPieChart";
import NivoLineChart from "../components/NivoLineChart";

import {
  Container,
  Paper,
  Button,
  Switch,
  Typography,
  FormControlLabel,
  FormGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@material-ui/core";

import Header from "../components/Header";
import Footer from "../components/Footer";

const EMOTION_SCALE = {
  happy: 30,
  suprised: 20,
  neutral: 15,
  disgust: 13,
  sad: 10,
  fear: 5,
  angry: 0,
};

class App extends React.Component {
  constructor(props) {
    super(props);
    this.intervalID = 0;
    this.sessionData = {};
    this.mostDetectedEmotionCounter = [];
    this.tempDissapearedFaces = [];
    this.state = {
      isWebcam: false,
      isButtonStart: false,
      isProcessing: false,
      startSession: false,
      currentVideo: "test_videos/video.1.mp4",
      frame: null,
      clearGlobals: false,
      xPoint: 0,
      pieChartData: [
        {
          id: "angry",
          label: "Angry",
          value: 0,
          color: "hsl(0, 70%, 50%)",
        },
        {
          id: "disgust",
          label: "Disgust",
          value: 0,
          color: "hsl(120, 70%, 50%)",
        },
        {
          id: "fear",
          label: "Fear",
          value: 0,
          color: "hsl(0, 0%, 0%)",
        },
        {
          id: "happy",
          label: "Happy",
          value: 0,
          color: "hsl(180, 70%, 50%)",
        },
        {
          id: "sad",
          label: "Sad",
          value: 0,
          color: "hsl(240, 70%, 50%)",
        },
        {
          id: "suprised",
          label: "Suprised",
          value: 0,
          color: "hsl(60, 70%, 50%)",
        },
        {
          id: "neutral",
          label: "Neutral",
          value: 0,
          color: "hsl(255, 0%, 60%)",
        },
      ],
      lineChartData: []
    };
  }

  async componentDidMount() {
    await new Promise((r) => setTimeout(r, 2000));
    this.startInterval();
  }

  startInterval = () => {
    this.intervalID = setInterval(async () => {
      await this.detectEmotionsInFrame();
    }, 1000);
  };

  stopInterval = () => {
    clearInterval(this.intervalID);
  };

  resetInterval = (param) => {
    this.stopInterval();
    this.setState({ isProcessing: true });
    setTimeout(() => {
      this.setState(param, () => {
        this.startInterval();
        this.setState({ isProcessing: false });
      });
    }, 1000);
  };

  startSession = () => {
    this.sessionData = {
      sessionStartTimestamp: new Date().toLocaleString(),
      duration: 0,
      maxDetectedFace: 0,
      mostDetecedEmotion: null,
    };
    this.mostDetectedEmotionCounter = [
      { emotion: "angry", counter: 0 },
      { emotion: "disgust", counter: 0 },
      { emotion: "fear", counter: 0 },
      { emotion: "happy", counter: 0 },
      { emotion: "sad", counter: 0 },
      { emotion: "suprised", counter: 0 },
      { emotion: "neutral", counter: 0 },
    ];

    this.setState({ startSession: true });
  };

  stopSession = () => {
    this.setState({ startSession: false });

    this.sessionData = {
      ...this.sessionData,
      mostDetecedEmotion: this.mostDetectedEmotionCounter.find(
        (mde) =>
          mde.counter ===
          Math.max.apply(
            Math,
            this.mostDetectedEmotionCounter.map((m) => m.counter)
          )
      ),
    };

    axios
      .post("http://127.0.0.1:5000/api/save-session-data", this.sessionData)
      .then((response) => {
        console.log(response);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  detectEmotionsInFrame = async () => {
    try {
      const capturedFrame = this.state.isWebcam
        ? this.webcam.getScreenshot()
        : captureVideoFrame(this.player.getInternalPlayer());
      console.log("[INFO] Sending the captured frame...");
      const response = await axios.post(
        "http://127.0.0.1:5000/api/emotion-detection",
        {
          capturedFrame: this.state.isWebcam
            ? capturedFrame
            : capturedFrame.dataUri,
          clearGlobals: this.state.clearGlobals,
        }
      );
      console.log("[INFO] Response returned.");

      if (response.data.error) return;

      const detectedEmotions = response.data.detectedEmotionsOnFaces;
      console.log(JSON.stringify(detectedEmotions));

      const dissapearedFaces = response.data.dissapearedFaces;
      console.log(JSON.stringify(dissapearedFaces));

      let pieChartData = JSON.parse(JSON.stringify(this.state.pieChartData));
      pieChartData.forEach((e) => (e.value = 0));
      detectedEmotions.forEach((de) => {
        pieChartData.find((e) => e.id === de.emotion).value += 1;
      });

      let lineChartData = JSON.parse(JSON.stringify(this.state.lineChartData));
      let xPoint = this.state.xPoint;
      if (this.state.clearGlobals) {
        lineChartData = [];
        this.tempDissapearedFaces = [];
        xPoint = 0;
      }

      if (xPoint >= 15) {
        lineChartData = lineChartData.map((lcd) => {
          if (lcd.data.length >= 15)
            lcd = {
              ...lcd,
              data: lcd.data.slice(1),
            };
          return lcd;
        });
      }

      this.tempDissapearedFaces.forEach((tdf) => {
        if (!dissapearedFaces.find((df) => df.id === tdf)) {
          lineChartData = lineChartData.filter((lcd) => lcd.id !== `ID ${tdf}`);
        }
      });
      this.tempDissapearedFaces = [];

      dissapearedFaces.forEach((df) => {
        const detectedEmo = detectedEmotions.find((de) => de.id === df.id);
        if (detectedEmo) {
          let chartData = lineChartData.find(
            (lcd) => lcd.id === `ID ${detectedEmo.id}`
          );
          let chartDataPoint = {
            x: xPoint,
            y: Math.round(
              (EMOTION_SCALE[detectedEmo.emotion] * 100) /
                Math.max(...Object.values(EMOTION_SCALE))
            ),
          };
          if (chartData) {
            chartData.data.push(chartDataPoint);
          } else {
            lineChartData.push({
              id: `ID ${detectedEmo.id}`,
              data: [chartDataPoint],
            });
          }
        } else {
          if (df.counter === 10) {
            this.tempDissapearedFaces.push(df.id);
          } else {
            let chartData = lineChartData.find(
              (lcd) => lcd.id === `ID ${df.id}`
            );
            chartData.data.push({
              x: xPoint,
              y: null,
            });
          }
        }
      });

      if (lineChartData.length > 0) {
        let avgData = lineChartData.find((lcd) => lcd.id === "Avarage");
        let avgDataPoint = {
          x: xPoint,
          y: Math.round(
            lineChartData
              .map((lcd) => lcd.data[lcd.data.length - 1].y)
              .reduce((acc, cur) => acc + cur) / lineChartData.length
          ),
        };
        if (avgData) {
          lineChartData = lineChartData.filter((lcd) => lcd.id !== "Avarage");
          avgData.data.push(avgDataPoint);
          lineChartData.push(avgData);
        } else {
          lineChartData.push({
            id: "Avarage",
            data: [avgDataPoint],
          });
        }
      }

      if (this.state.startSession) {
        this.mostDetectedEmotionCounter.forEach((mde) => {
          mde.counter += this.state.pieChartData.find(
            (pcd) => pcd.id === mde.emotion
          ).value;
        });

        this.sessionData = {
          ...this.sessionData,
          duration: this.sessionData.duration + 1,
          maxDetectedFace:
            detectedEmotions.length > this.sessionData.maxDetectedFace
              ? detectedEmotions.length
              : this.sessionData.maxDetectedFace,
        };
      }

      if (this.state.clearGlobals) this.setState({ clearGlobals: false });
      this.setState({
        frame: "data:image/jpg;base64," + response.data.frame,
        pieChartData: pieChartData,
        lineChartData: lineChartData,
        xPoint: xPoint + 1,
      });
    } catch (error) {
      console.log("[ERROR] " + error.stack);
    }
  };

  render() {
    return (
      <div style={{ backgroundColor: "#F6F6F6" }}>
        <Header />
        <Container
          maxWidth="xl"
          style={{ marginTop: "20px", marginBottom: "20px" }}
        >
          <Paper
            elevation={0}
            style={{ padding: "20px", backgroundColor: "#F6F6F6" }}
          >
            <Container
              maxWidth="xl"
              style={{
                display: "flex",
                flexWrap: "wrap",
                justifyContent: "space-evenly",
                backgroundColor: "#F6F6F6",
              }}
            >
              <Paper elevation={3} style={{ padding: "20px" }}>
                <Typography variant="h4" gutterBottom>
                  {this.state.isWebcam ? "Webcam" : "Video"}
                </Typography>
                {this.state.isWebcam ? (
                  <ReactWebcam
                    audio={false}
                    screenshotFormat="image/jpeg"
                    ref={(node) => (this.webcam = node)}
                  />
                ) : (
                  <ReactPlayer
                    url={this.state.currentVideo}
                    playing={true}
                    loop={true}
                    ref={(player) => (this.player = player)}
                  />
                )}
              </Paper>
              <Paper elevation={3} style={{ padding: "20px" }}>
                <Typography variant="h4" gutterBottom>
                  Detected Emotion(s)
                </Typography>
                {this.state.frame && <img src={this.state.frame} alt="" />}
              </Paper>
            </Container>
            <Container
              maxWidth="xl"
              style={{
                marginTop: "30px",
              }}
            >
              <Paper elevation={3} style={{ padding: "20px" }}>
                <Typography variant="h5" style={{ marginBottom: "15px" }}>
                  Control Panel
                </Typography>
                <FormControl
                  component="fielset"
                  style={{ marginLeft: "10px" }}
                  disabled={this.state.isProcessing}
                >
                  <FormGroup row>
                    <FormControlLabel
                      control={
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={() => {
                            this.state.isButtonStart
                              ? this.startInterval()
                              : this.stopInterval();
                            this.setState({
                              isButtonStart: !this.state.isButtonStart,
                            });
                          }}
                        >
                          {this.state.isButtonStart ? "Start" : "Stop"}
                        </Button>
                      }
                    />
                    <FormControlLabel
                      control={
                        <Button
                          variant="contained"
                          color="secondary"
                          onClick={() => {
                            this.resetInterval({
                              clearGlobals: true,
                              isButtonStart: false,
                            });
                          }}
                        >
                          Reset
                        </Button>
                      }
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={this.state.isWebcam}
                          onChange={() => {
                            this.resetInterval({
                              isWebcam: !this.state.isWebcam,
                              clearGlobals: true,
                              isButtonStart: false,
                            });
                          }}
                        />
                      }
                      label="Webcam"
                    />
                    {this.state.isWebcam ? null : (
                      <FormControl
                        variant="outlined"
                        style={{ width: "150px" }}
                        disabled={this.state.isProcessing}
                      >
                        <InputLabel>Select Video</InputLabel>
                        <Select
                          value={this.state.currentVideo}
                          label="Select Video"
                          onChange={(e) => {
                            this.resetInterval({
                              clearGlobals: true,
                              isButtonStart: false,
                              currentVideo: e.target.value,
                            });
                          }}
                        >
                          <MenuItem value={"test_videos/video.1.mp4"}>
                            Video 1
                          </MenuItem>
                          <MenuItem value={"test_videos/video.2.mp4"}>
                            Video 2
                          </MenuItem>
                          <MenuItem value={"test_videos/video.3.mp4"}>
                            Video 3
                          </MenuItem>
                          <MenuItem value={"test_videos/video.4.mp4"}>
                            Video 4
                          </MenuItem>
                          <MenuItem value={"test_videos/video.5.mp4"}>
                            Video 5
                          </MenuItem>
                          <MenuItem value={"test_videos/video.6.mp4"}>
                            Video 6
                          </MenuItem>
                          <MenuItem value={"test_videos/video.7.mp4"}>
                            Video 7
                          </MenuItem>
                          <MenuItem value={"test_videos/video.8.mp4"}>
                            Video 8
                          </MenuItem>
                        </Select>
                      </FormControl>
                    )}
                    <FormControlLabel
                      style={{ marginLeft: "15px" }}
                      control={
                        <Button
                          variant="contained"
                          color="secondary"
                          onClick={() => {
                            this.state.startSession
                              ? this.stopSession()
                              : this.startSession();
                          }}
                        >
                          {!this.state.startSession ? "Start" : "Stop and Save"}{" "}
                          Session
                        </Button>
                      }
                    />
                  </FormGroup>
                </FormControl>
              </Paper>
            </Container>
            <Container
              maxWidth="xl"
              style={{
                marginTop: "30px",
                display: "flex",
                flexWrap: "wrap",
                justifyContent: "space-around",
              }}
            >
              <Paper
                elevation={3}
                style={{ padding: "20px", marginRight: "20px" }}
              >
                <Typography variant="h4">Satisfaction Chart</Typography>
                <NivoLineChart data={this.state.lineChartData} />
              </Paper>
              <Paper elevation={3} style={{ padding: "20px" }}>
                <Typography variant="h4">Emotions Chart</Typography>
                <NivoPieChart data={this.state.pieChartData} />
              </Paper>
            </Container>
          </Paper>
        </Container>
        <Footer />
      </div>
    );
  }
}
export default App;
