
const Parser = require('web-tree-sitter');

(async () => {
  await Parser.init();
  const parser = new Parser();
  const Lang = await Parser.Language.load('tree-sitter-q.wasm');
  parser.setLanguage(Lang);
  const tree = parser.parse('2*2');
  console.log(tree.rootNode.toString());
  console.log(JSON.stringify(tree.rootNode))
})();