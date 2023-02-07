const { NodeSSH } = require("node-ssh");
const fs = require("fs");

const ssh = new NodeSSH();

let sshConfig = {
  host: "localhost",
  username: "pi",
  password: "",
};

let shellStream = null;
const executeCommand = (comm) => {
  if (shellStream) {
    shellStream.write(comm);
  }
};
const onConnect = async (ws, req) => {
  try {
    const model = fs.readFileSync("/proc/device-tree/model", "utf8").trim();
    if (model.includes("Jetson Nano")) {
      sshConfig.password = "pi";
    } else if (model.includes("Raspberry Pi") || model.includes("Nano")) {
      sshConfig.password = "raspberry";
    } else {
      sshConfig.password = "raspberry";
    }
  } catch (err) {
    console.error(`Error while reading /proc/device-tree/model: ${err}`);
  }
  await ssh.connect(sshConfig);
  shellStream = await ssh.requestShell();
  ws.on("message", (msg) => {
    if (msg == "CMD:TERM") {
      shellStream.write("\x03");
      shellStream.write("\x03");
      shellStream.write("\n");
      shellStream.write("\n");
    } else {
      shellStream.write(msg);
    }
  });
  // listener
  shellStream.on("data", (data) => {
    ws.send(data.toString());
  });
  shellStream.stderr.on("data", (data) => {
    ws.send(d.toString());
  });
};

module.exports.connection = onConnect;
module.exports.command = executeCommand;
