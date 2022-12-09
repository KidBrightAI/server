const fastify = require("fastify");
const path = require("path");
const fs = require("fs");
const cors = require("@fastify/cors");
const app = fastify();
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
app.listen(8888, "0.0.0.0");
