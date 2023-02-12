import Parser from "web-tree-sitter";
//import SyntaxNode from "web-tree-sitter";
//import Language from "tree-sitter";

// import ts from 'ts-transformer-keys';
//
// function freeze<Interface extends object>(instance :Interface) :Interface {
//     let output :Interface = {} as any;
//     for (let property of ts.keys<Interface>())
//         output[property] = instance[property];
//     return output;
// }

await Parser.init();

const parser = new Parser();
const lang = await Parser.Language.load('tree-sitter-q.wasm');
parser.setLanguage(lang);

//const tree: Parser.Tree = parser.parse('factorial:{$[x<2;1;x*.z.s x-1]}');
const tree: Parser.Tree = parser.parse('1+3');
//const tree: Parser.Tree = parser.parse('+;2;3');
console.log(tree.rootNode.toString());
const sn: Parser.SyntaxNode = tree.rootNode;

//console.log(JSON.stringify(sn, Object.keys(SyntaxNode.prototype)));
console.log(toJSON(sn));

function toJSON(sn: Parser.SyntaxNode) {
    return {type: sn.type, childCount: sn.childCount}
}

console.log(JSON.stringify(sn, function(key, value) {
      if (typeof value === 'function') {
        return value.toString();
      } else {
        return value;
      }
  }, "  "));







