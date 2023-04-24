const fastify = require("fastify");
const path = require("path");
const fs = require("fs");
const cors = require("@fastify/cors");
const app = fastify();

const wifi = require("node-wifi");
wifi.init({
  iface: null, // network interface, choose a random wifi interface if set to null
});

app.register(require("fastify-ws"));
app.register(cors);
const WsController = require("./controller/WebConsole");

app.ready((err) => {
  if (err) throw err;
  console.log("server started...");
  app.ws.on("connection", WsController.connection);
});
app.get("/", (req, res) => {
  res.send({ hello: "word" });
});
app.post("/run", (req, res) => {
  const projectId = req.body.project_id;
  const code = Buffer.from(req.body.code, "base64").toString("utf8");
  const projectPath = path.join("../projects", projectId);
  const targetFile = path.join(projectPath, "run.py");
  fs.writeFileSync(targetFile, code);
  WsController.command("python3 " + path.resolve(targetFile) + "\r\n");
  res.send({ success: true });
});

app.post("/wifiConnect", (req, res, next) => {
  console.log("Connecting to wifi");
  console.log(req.body);
  wifi
    .connect({ ssid: req.body.ssid, password: req.body.password })
    .then(() => {
      console.log("Connected to network.");
      res.send(
        JSON.stringify({
          status: "OK",
        })
      );
    })
    .catch((error) => {
      console.log(error);
    });
});

app.get("/listWifi", function (req, res, next) {
  console.log("list wifi");
  wifi
    .scan()
    .then((networks) => {
      console.log(networks);
      ssids = [];
      for (var i = 0; i < networks.length; i++) {
        console.log(networks[i]);
        ssids.push(networks[i].ssid);
      }

      console.log(ssids);

      res.send(
        JSON.stringify({
          ap: ssids,
          status: "OK",
        })
      );
    })
    .catch((error) => {
      console.log(error);
    });
});

app.listen(8888, "0.0.0.0");
