
const Parser = require('web-tree-sitter');

(async () => {
  await Parser.init();
  const parser = new Parser();
  const Lang = await Parser.Language.load('tree-sitter-q.wasm');
  parser.setLanguage(Lang);
  const tree = parser.parse('factorial:{$[x<2;1;x*.z.s x-1]}');
  console.log(tree.rootNode.toString());
  console.log(tree.rootNode);
  console.log(tree.rootNode.type);
  //console.log(JSON.stringify(tree.rootNode, null, "  "));

  console.log(JSON.stringify(tree.rootNode, function(key, value) {
      if (typeof value === 'function') {
        return value.toString();
      } else {
        return value;
      }
  }, "  "));

  const expr = tree.rootNode.child(0);
  console.log(expr.type);
  console.log(expr);
})();