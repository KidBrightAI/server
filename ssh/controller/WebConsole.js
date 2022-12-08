const { NodeSSH } = require("node-ssh");
const ssh = new NodeSSH();

const sshConfig = {
  host: "localhost",
  username: "pi",
  password: "raspberry",
};
let shellStream = null;
const executeCommand = (comm) => {
  if (shellStream) {
    shellStream.write(comm);
  }
};
const onConnect = async (ws, req) => {
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
