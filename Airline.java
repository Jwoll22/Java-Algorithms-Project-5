import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.ArrayList;
import java.io.*;

public class Airline {
    private double[] distTo;          // distTo[v] = distance  of shortest s->v path
    private Edge[] edgeTo;    // edgeTo[v] = last edge on shortest s->v path
    private boolean[] marked;     // marked[v] = true if v on tree, false otherwise
    private IndexMinPQ<Double> pq;    // priority queue of vertices
    private Bag<Edge>[] adj;
    private boolean boolDist;
    private EdgeWeightedGraph graph;
    private ArrayList<String> cities;

    public Airline(String filename) {
        try {
            File file = new File(filename);
            BufferedReader br = new BufferedReader(new FileReader(file));
            graph = new EdgeWeightedGraph(br);

            PrimMSTTrace mst = new PrimMSTTrace(graph);
            System.out.println("MINIMUM SPANNING TREE");
            System.out.println("---------------------");
            System.out.println("The edges in the MST based on distance follow:");
            for (Edge e : mst.edges2()) {
                System.out.println(e.name + " : " + e.distance);
            }
            System.out.println();

            //Program
            boolean done = false;
            BufferedReader reader =
                       new BufferedReader(new InputStreamReader(System.in));
            while (!done) {
                System.out.println("What do you want?");
                System.out.println("a) Shortest Path");
                System.out.println("b) Find trips by cost");
                System.out.println("c) Add a route");
                System.out.println("d) Remove a route");
                System.out.println("e) Exit");
                String selection = reader.readLine();
                switch (selection) {
                    //Shortest paths
                    case "a": System.out.println("Filter by what?");
                                    System.out.println("a) Distance");
                                    System.out.println("b) Price");
                                    System.out.println("c) Hops");
                                    String choice = reader.readLine();
                                    switch (choice) {
                                        case "a": System.out.println("Start where?");
                                                        String v1 = reader.readLine();
                                                        System.out.println("End where?");
                                                        String v2 = reader.readLine();
                                                        boolDist = true;
                                                        Dijkstra(boolDist, cities.indexOf(v1), cities.indexOf(v2));
                                                        break;
                                        case "b": System.out.println("Start where?");
                                                        String v3 = reader.readLine();
                                                        System.out.println("End where?");
                                                        String v4 = reader.readLine();
                                                        boolDist = false;
                                                        Dijkstra(boolDist, cities.indexOf(v3), cities.indexOf(v4));
                                                        break;
                                        case "c": System.out.println("Start where?");
                                                        String v5 = reader.readLine();
                                                        System.out.println("End where?");
                                                        String v6 = reader.readLine();
                                                        BFS(cities.indexOf(v5), cities.indexOf(v6));
                                                        break;
                                    }
                                    break;
                    //Price filter
                    case "b": System.out.println("Enter the maximum price for a trip");
                                    double bound = Double.parseDouble(reader.readLine());
                                    System.out.println("ALL PATHS OF COST " + bound + " OR LESS");
                                    System.out.println("Note that paths are duplicated, once from each end city's point of view");
                                    System.out.println("-----------------------------------------------------------------------");
                                    System.out.println("List of paths at most " + bound + " in length: ");
                                    boolDist = false;
                                    for (int i = 0; i < graph.V; i++) {
                                        DijkstraSP sp = new DijkstraSP(graph, i);
                                        for (int j = 0; j < graph.V && j != i; j++) {
                                            if (sp.hasPathTo(j) && sp.distTo(j) <= bound) {
                                                System.out.print("Cost: " + sp.distTo(j) + "  ");
                                                for (Edge e : sp.pathTo(j)) {
                                                    System.out.print(e.name + " " + e.price + " ");
                                                }
                                                System.out.println();
                                            }
                                        }
                                    }
                                    System.out.println();
                                    break;
                    //Add
                    case "c": System.out.println("Start where?");
                                    String v1 = reader.readLine();
                                    System.out.println("End where?");
                                    String v2 = reader.readLine();
                                    System.out.println("What is the distance?");
                                    int newDist = Integer.parseInt(reader.readLine());
                                    double newPrice = Double.parseDouble(reader.readLine());
                                    Edge e = new Edge(cities.indexOf(v1),cities.indexOf(v2), newDist, newPrice, v1+","+v2);
                                    Edge eReverse = new Edge(cities.indexOf(v2),cities.indexOf(v1), newDist, newPrice, v2+","+v1);
                                    graph.addEdge(e);
                                    graph.addEdge(eReverse);
                                    System.out.println();
                                    break;
                    //Remove
                    case "d": System.out.println("Start where?");
                                    String v3 = reader.readLine();
                                    System.out.println("End where?");
                                    String v4 = reader.readLine();
                                    graph.removeEdge(findEdge(v3,v4));
                                    graph.removeEdge(findEdge(v4,v3));
                                    System.out.println();
                                    break;
                    case "e": done = true;
                                    System.out.println("Exiting....");
                                    break;
                }
            }
        }
        catch (Exception e) {
            System.err.println("Exception: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.println("INPUT FILE: " + args[0]);
        System.out.println("---------------------");
        System.out.println();
        Airline yeet = new Airline(args[0]);
    }

    public void Dijkstra(boolean weight, int start, int end) {
        boolDist = weight;
        DijkstraSP sp = new DijkstraSP(graph, start);
        if (sp.hasPathTo(end)) {
            if (weight) {
                System.out.println("SHORTEST DISTANCE PATH from " + cities.get(start) + " to " + cities.get(end));
                System.out.println("--------------------------------------------------");
                System.out.println("Shortest distance from " + cities.get(start) + " to " + cities.get(end) + " is " + sp.distTo(end));
                for (Edge e : sp.pathTo(end)) {
                    System.out.print(e.name + " " + e.distance + " ");
                }
                System.out.println();
            }
            else {
                System.out.println("SHORTEST COST PATH from " + cities.get(start) + " to " + cities.get(end));
                System.out.println("----------------------------------------------");
                System.out.println("Shortest cost from " + cities.get(start) + " to " + cities.get(end) + " is " + sp.distTo(end));
                for (Edge e : sp.pathTo(end)) {
                    System.out.print(e.name + " " + e.distance + " ");
                }
                System.out.println();
            }
        }
        else
            System.out.println("There exists no path for these two vertices");
    }

    public void BFS(int start, int end) {
        BreadthFirstPaths bfp = new BreadthFirstPaths(graph, start);
        if (bfp.hasPathTo(end)) {
            System.out.println("FEWEST HOPS from " + cities.get(start) + " to " + cities.get(end));
            System.out.println("---------------------------------------");
            System.out.println("Fewest hops from " + cities.get(start) + " to " + cities.get(end) + " is " + bfp.distTo(end));
            for (Edge e : bfp.pathTo(end)) {
                System.out.print(e.name + " ");
            }
            System.out.println();
        }
    }

    public Edge findEdge(String v1, String v2) {
        if (!cities.contains(v1) && !cities.contains(v2)) return null;
        int v = cities.indexOf(v1);
        int w = cities.indexOf(v2);
        for (Edge e: graph.edges()) {
            if (v == e.either()) {
                if (w == e.other(v))
                    return e;
            }
        }
        return null;
    }

    /****************************
    * Graph Implementation
    ****************************/
    private class EdgeWeightedGraph {
        private int V;
        private int E;
       /**
         * Create a weighted graph from input stream.
         */
         @SuppressWarnings("unchecked")
        public EdgeWeightedGraph(BufferedReader in) {
            try {
                this.V = Integer.parseInt(in.readLine());
                if (V < 0) throw new RuntimeException("Number of vertices must be nonnegative");
                this.E = 0;
                adj = (Bag<Edge>[]) new Bag<?>[V];
                for (int v = 0; v < V; v++) adj[v] = new Bag<Edge>();
                cities = new ArrayList<String>();
                for (int i = 0; i < V; i++) {
                    cities.add(in.readLine());
                    //System.out.println(cities[i]);
                }
                String line;
                while ((line = in.readLine()) != null) {
                    String[] edgeData = line.split(" ");
                    int v = Integer.parseInt(edgeData[0]);
                    //System.out.println("v: " + v);
                    int w = Integer.parseInt(edgeData[1]);
                    //System.out.println("w: " + w);
                    int distance = Integer.parseInt(edgeData[2]);
                    //System.out.println("Distance: " + distance);
                    double price = Double.parseDouble(edgeData[3]);
                    Edge e = new Edge(v-1, w-1, distance, price, cities.get(v-1) + "," + cities.get(w-1));
                    Edge eReverse = new Edge(w-1, v-1, distance, price, cities.get(w-1) + "," + cities.get(v-1));
                    addEdge(e);
                    addEdge(eReverse);
                    this.E++;
                    line = "";
                }
            }
            catch (Exception e) {
                System.err.println("Exception: " + e.getMessage());
            }
        }

       /**
         * Return the number of vertices in this graph.
         */
        public int V() {
            return V;
        }

       /**
         * Return the number of edges in this graph.
         */
        public int E() {
            return E;
        }

       /**
         * Add the edge e to this graph.
         */
        public void addEdge(Edge e) {
            int v = e.either();
            int w = e.other(v);
            adj[v].add(e);
            adj[w].add(e);
            E++;
        }

        public void removeEdge(Edge e) {
            int v = e.either();
            int w = e.other(v);
            adj[v].remove(e);
            adj[w].remove(e);
        }

       /**
         * Return the edges incident to vertex v as an Iterable.
         * To iterate over the edges incident to vertex v, use foreach notation:
         * <tt>for (Edge e : graph.adj(v))</tt>.
         */
        public Iterable<Edge> adj(int v) {
            return adj[v];
        }

       /**
         * Return all edges in this graph as an Iterable.
         * To iterate over the edges, use foreach notation:
         * <tt>for (Edge e : graph.edges())</tt>.
         */
        public Iterable<Edge> edges() {
            Bag<Edge> list = new Bag<Edge>();
            for (int v = 0; v < V; v++) {
                int selfLoops = 0;
                for (Edge e : adj(v)) {
                    if (e.other(v) > v) {
                        list.add(e);
                    }
                    // only add one copy of each self loop
                    else if (e.other(v) == v) {
                        if (selfLoops % 2 == 0) list.add(e);
                        selfLoops++;
                    }
                }
            }
            return list;
        }

       /**
         * Return a string representation of this graph.
         */
        public String toString() {
            String NEWLINE = System.getProperty("line.separator");
            StringBuilder s = new StringBuilder();
            s.append(V + " " + E + NEWLINE);
            for (int v = 0; v < V; v++) {
                s.append(v + ": ");
                for (Edge e : adj[v]) {
                    s.append(e + "  ");
                }
                s.append(NEWLINE);
            }
            return s.toString();
        }
    }

    /****************************
    * Eager Prim's methods
    ****************************/
    private class PrimMSTTrace {
        public PrimMSTTrace(EdgeWeightedGraph G) {
            edgeTo = new Edge[G.V()];
            distTo = new double[G.V()];
            marked = new boolean[G.V()];
            pq = new IndexMinPQ<Double>(G.V());
            for (int v = 0; v < G.V(); v++) distTo[v] = Double.POSITIVE_INFINITY;

            for (int v = 0; v < G.V(); v++)      // run from each vertex to find
                if (!marked[v]) prim(G, v);      // minimum spanning forest
        }

        // run Prim's algorithm in graph G, starting from vertex s
        private void prim(EdgeWeightedGraph G, int s) {
            distTo[s] = 0.0;
            pq.insert(s, distTo[s]);
            //showPQ(pq);
            while (!pq.isEmpty()) {
                int v = pq.delMin();
                //System.out.println("	Next Vertex (Weight): " + v + " (" + distTo[v] + ")");
                scan(G, v);
                //showPQ(pq);
            }
        }

        // scan vertex v
        private void scan(EdgeWeightedGraph G, int v) {
            marked[v] = true;
            //System.out.println("	Checking neighbors of " + v);
            for (Edge e : G.adj(v)) {
                double weight = e.weight(true);
                int w = e.other(v);
                //System.out.print("		Neighbor " + w);
                if (marked[w])
                {
                	//System.out.println(" is in the tree ");
                	continue;         // v-w is obsolete edge
                }
                if (weight < distTo[w]) {
                	//System.out.print(" OLD distance: " + distTo[w]);
                    distTo[w] = weight;
                    edgeTo[w] = e;
                    //System.out.println(" NEW distance: " + distTo[w]);
                    if (pq.contains(w))
                    {
                    		pq.change(w, distTo[w]);
                    		//System.out.println("			PQ changed");
                    }
                    else
                    {
                    		pq.insert(w, distTo[w]);
                    		//System.out.println("			Inserted into PQ");
                    }
                }
                //else
                	//System.out.println(" distance " + distTo[w] + " NOT CHANGED");
            }
        }

        // return iterator of edges in MST
        public Iterable<Edge> edges2() {
            Bag<Edge> mst = new Bag<Edge>();
            for (int v = 0; v < edgeTo.length; v++) {
                Edge e = edgeTo[v];
                if (e != null) {
                    mst.add(e);
                }
            }
            return mst;
        }

        // return weight of MST
        public double totWeight() {
            double weight = 0.0;
            for (Edge e : edges2())
                weight += e.weight(true);
            return weight;
        }
    }

    /****************************
    * Dijkstra's methods
    ****************************/
    private class DijkstraSP {
        public DijkstraSP(EdgeWeightedGraph G, int s) {
            distTo = new double[G.V()];
            edgeTo = new Edge[G.V()];
            for (int v = 0; v < G.V(); v++)
                distTo[v] = Double.POSITIVE_INFINITY;
            distTo[s] = 0.0;

            // relax vertices in order of distance from s
            pq = new IndexMinPQ<Double>(G.V());
            pq.insert(s, distTo[s]);
            while (!pq.isEmpty()) {
                int v = pq.delMin();
                for (Edge e : G.adj(v))
                    relax(e);
            }
        }

        // relax edge e and update pq if changed
        private void relax(Edge e) {
            double weight = e.weight(boolDist);
            int v = e.either(), w = e.other(v);
            if (distTo[w] > distTo[v] + weight) {
                distTo[w] = distTo[v] + weight;
                edgeTo[w] = e;
                if (pq.contains(w)) pq.change(w, distTo[w]);
                else                pq.insert(w, distTo[w]);
            }
        }

        // length of shortest path from s to v
        public double distTo(int v) {
            return distTo[v];
        }

        // is there a path from s to v?
        public boolean hasPathTo(int v) {
            return distTo[v] < Double.POSITIVE_INFINITY;
        }

        // shortest path from s to v as an Iterable, null if no such path
        public Iterable<Edge> pathTo(int v) {
            if (!hasPathTo(v)) return null;
            Stack<Edge> path = new Stack<Edge>();
            for (Edge e = edgeTo[v]; e != null; e = edgeTo[e.either()]) {
                path.push(e);
            }
            return path;
        }
    }

    /****************************
    *Breadth First Paths Subclass
    ****************************/
    public class BreadthFirstPaths {
        private static final int INFINITY = Integer.MAX_VALUE;
        private int[] distTo;      // distTo[v] = number of edges shortest s-v path
        private final int s;       // source vertex
        public BreadthFirstPaths(EdgeWeightedGraph G, int s) {
            marked = new boolean[G.V()];
            distTo = new int[G.V()];
            edgeTo = new Edge[G.V()];
            this.s = s;
            bfs(G, s);
        }

        private void bfs(EdgeWeightedGraph G, int s) {
            Queue<Integer> q = new Queue<Integer>();
            for (int v = 0; v < G.V(); v++) distTo[v] = INFINITY;
            distTo[s] = 0;
            marked[s] = true;
            q.enqueue(s);

            while (!q.isEmpty()) {
                int v = q.dequeue();
                for (Edge e : G.adj(v)) {
                    int w = e.other(v);
                    if (!marked[w]) {
                        edgeTo[w] = e;
                        distTo[w] = distTo[v] + 1;
                        marked[w] = true;
                        q.enqueue(w);
                    }
                }
            }
        }

        public boolean hasPathTo(int v) {
            return marked[v];
        }

        public int distTo(int v) {
            return distTo[v];
        }

        // return shortest path from s to v; null if no such path
        public Iterable<Edge> pathTo(int v) {
            if (!hasPathTo(v)) return null;
            Stack<Edge> path = new Stack<Edge>();
            for (Edge e = edgeTo[v]; e != null; e = edgeTo[e.either()]) {
                path.push(e);
            }
            return path;
        }
    }

    /****************************
    * Weighted Undirected Edge Subclass
    ****************************/
    private class Edge implements Comparable<Edge> {
         private final int v;
         private final int w;
         private final int distance;
         private final double price;
         private final String name;

        /**
          * Create an edge between v and w with given distance and price.
          */
         public Edge(int v, int w, int distance, double price, String name) {
             this.v = v;
             this.w = w;
             this.distance = distance;
             this.price = price;
             this.name = name;
         }

         /**
           * Return the distance of this edge.
           */
          public double weight(boolean dist) {
              if (dist) return distance;
              else  return price;
          }

        /**
          * Return either endpoint of this edge.
          */
         public int either() {
             return v;
         }

        /**
          * Return the endpoint of this edge that is different from the given vertex
          * (unless a self-loop).
          */
         public int other(int vertex) {
             if      (vertex == v) return w;
             else if (vertex == w) return v;
             else throw new RuntimeException("Illegal endpoint");
         }

         /**
           * Compare edges by dist or price.
           */
          public int compareTo(Edge that) {
              if      (this.weight(boolDist) < that.weight(boolDist)) return -1;
              else if (this.weight(boolDist) > that.weight(boolDist)) return +1;
              else                                    return  0;
          }

        /**
          * Return a string representation of this edge.
          */
         public String toString() {
             return String.format("%d-%d %.2f", v, w, distance, price, name);
         }
     }

     /**************************************************************
     *Bag Subclass
     **************************************************************/
     private class Bag<Item> implements Iterable<Item> {
        private int N;         // number of elements in bag
        private Node first;    // beginning of bag

        // helper linked list class
        private class Node {
            private Item item;
            private Node next;
        }

       /**
         * Create an empty stack.
         */
        public Bag() {
            first = null;
            N = 0;
        }

       /**
         * Is the BAG empty?
         */
        public boolean isEmpty() {
            return first == null;
        }

       /**
         * Return the number of items in the bag.
         */
        public int size() {
            return N;
        }

       /**
         * Add the item to the bag.
         */
        public void add(Item item) {
            Node oldfirst = first;
            first = new Node();
            first.item = item;
            first.next = oldfirst;
            N++;
        }

        public void remove(Item item) {
            if (first.item.equals(item))    first = first.next;
            else {
                Node temp = first;
                while (temp.next != null && !temp.next.item.equals(item))
                    temp = temp.next;
                if (temp.next.next != null) temp.next = temp.next.next;
                else    temp.next = null;
                N--;
            }
        }

       /**
         * Return an iterator that iterates over the items in the bag.
         */
        public Iterator<Item> iterator()  {
            return new LIFOIterator();
        }

        // an iterator, doesn't implement remove() since it's optional
        private class LIFOIterator implements Iterator<Item> {
            private Node current = first;

            public boolean hasNext()  { return current != null;                     }
            public void remove()      { throw new UnsupportedOperationException();  }

            public Item next() {
                if (!hasNext()) throw new NoSuchElementException();
                Item item = current.item;
                current = current.next;
                return item;
            }
        }
    }

    /****************************
    * Indexable MinPQ Subclass
    ****************************/
    private class IndexMinPQ<Key extends Comparable<Key>> implements Iterable<Integer> {
        private int N;           // number of elements on PQ
        private int[] pq;        // binary heap using 1-based indexing
        private int[] qp;        // inverse of pq - qp[pq[i]] = pq[qp[i]] = i
        private Key[] keys;      // keys[i] = priority of i

        @SuppressWarnings("unchecked")
        public IndexMinPQ(int NMAX) {
            keys = (Key[]) new Comparable<?>[NMAX + 1];    // make this of length NMAX??
            pq   = new int[NMAX + 1];
            qp   = new int[NMAX + 1];                   // make this of length NMAX??
            for (int i = 0; i <= NMAX; i++) qp[i] = -1;
        }

        // is the priority queue empty?
        public boolean isEmpty() { return N == 0; }

        // is k an index on the priority queue?
        public boolean contains(int k) {
            return qp[k] != -1;
        }

        // number of keys in the priority queue
        public int size() {
            return N;
        }

        // associate key with index k
        public void insert(int k, Key key) {
            if (contains(k)) throw new RuntimeException("item is already in pq");
            N++;
            qp[k] = N;
            pq[N] = k;
            keys[k] = key;
            swim(N);
        }

        // delete a minimal key and returns its associated index
        public int delMin() {
            if (N == 0) throw new RuntimeException("Priority queue underflow");
            int min = pq[1];
            exch(1, N--);
            sink(1);
            qp[min] = -1;            // delete
            keys[pq[N+1]] = null;    // to help with garbage collection
            pq[N+1] = -1;            // not needed
            return min;
        }

        // change the key associated with index k
        public void change(int k, Key key) {
            if (!contains(k)) throw new RuntimeException("item is not in pq");
            keys[k] = key;
            swim(qp[k]);
            sink(qp[k]);
        }


       /**************************************************************
        * General helper functions
        **************************************************************/
        private boolean greater(int i, int j) {
            return keys[pq[i]].compareTo(keys[pq[j]]) > 0;
        }

        private void exch(int i, int j) {
            int swap = pq[i]; pq[i] = pq[j]; pq[j] = swap;
            qp[pq[i]] = i; qp[pq[j]] = j;
        }


       /**************************************************************
        * Heap helper functions
        **************************************************************/
        private void swim(int k)  {
            while (k > 1 && greater(k/2, k)) {
                exch(k, k/2);
                k = k/2;
            }
        }

        private void sink(int k) {
            while (2*k <= N) {
                int j = 2*k;
                if (j < N && greater(j, j+1)) j++;
                if (!greater(k, j)) break;
                exch(k, j);
                k = j;
            }
        }


       /***********************************************************************
        * Iterators
        **********************************************************************/

       /**
         * Return an iterator that iterates over all of the elements on the
         * priority queue in ascending order.
         * <p>
         * The iterator doesn't implement <tt>remove()</tt> since it's optional.
         */
        public Iterator<Integer> iterator() { return new HeapIterator(); }

        private class HeapIterator implements Iterator<Integer> {
            // create a new pq
            private IndexMinPQ<Key> copy;

            // add all elements to copy of heap
            // takes linear time since already in heap order so no keys move
            public HeapIterator() {
                copy = new IndexMinPQ<Key>(pq.length - 1);
                for (int i = 1; i <= N; i++)
                    copy.insert(pq[i], keys[pq[i]]);
            }

            public boolean hasNext()  { return !copy.isEmpty();                     }
            public void remove()      { throw new UnsupportedOperationException();  }

            public Integer next() {
                if (!hasNext()) throw new NoSuchElementException();
                return copy.delMin();
            }
        }
    }

    /****************************
    * Stack Subclass
    ****************************/
    private class Stack<Item> implements Iterable<Item> {
        private int N;          // size of the stack
        private Node first;     // top of stack

        // helper linked list class
        private class Node {
            private Item item;
            private Node next;
        }

       /**
         * Create an empty stack.
         */
        public Stack() {
            first = null;
            N = 0;
        }

       /**
         * Is the stack empty?
         */
        public boolean isEmpty() {
            return first == null;
        }

       /**
         * Return the number of items in the stack.
         */
        public int size() {
            return N;
        }

       /**
         * Add the item to the stack.
         */
        public void push(Item item) {
            Node oldfirst = first;
            first = new Node();
            first.item = item;
            first.next = oldfirst;
            N++;
        }

       /**
         * Return string representation.
         */
        public String toString() {
            StringBuilder s = new StringBuilder();
            for (Item item : this)
                s.append(item + " ");
            return s.toString();
        }


       /**
         * Return an iterator to the stack that iterates through the items in LIFO order.
         */
        public Iterator<Item> iterator()  { return new LIFOIterator();  }

        // an iterator, doesn't implement remove() since it's optional
        private class LIFOIterator implements Iterator<Item> {
            private Node current = first;
            public boolean hasNext()  { return current != null;                     }
            public void remove()      { throw new UnsupportedOperationException();  }

            public Item next() {
                if (!hasNext()) throw new NoSuchElementException();
                Item item = current.item;
                current = current.next;
                return item;
            }
        }
    }

    /****************************
    * Queue Find Subclass
    ****************************/
    public class Queue<Item> implements Iterable<Item> {
        private int N;         // number of elements on queue
        private Node first;    // beginning of queue
        private Node last;     // end of queue

        // helper linked list class
        private class Node {
            private Item item;
            private Node next;
        }

       /**
         * Create an empty queue.
         */
        public Queue() {
            first = null;
            last  = null;
        }

       /**
         * Is the queue empty?
         */
        public boolean isEmpty() {
            return first == null;
        }

       /**
         * Return the number of items in the queue.
         */
        public int size() {
            return N;
        }

       /**
         * Return the item least recently added to the queue.
         * Throw an exception if the queue is empty.
         */
        public Item peek() {
            if (isEmpty()) throw new RuntimeException("Queue underflow");
            return first.item;
        }

       /**
         * Add the item to the queue.
         */
        public void enqueue(Item item) {
            Node x = new Node();
            x.item = item;
            if (isEmpty()) { first = x;     last = x; }
            else           { last.next = x; last = x; }
            N++;
        }

       /**
         * Remove and return the item on the queue least recently added.
         * Throw an exception if the queue is empty.
         */
        public Item dequeue() {
            if (isEmpty()) throw new RuntimeException("Queue underflow");
            Item item = first.item;
            first = first.next;
            N--;
            if (isEmpty()) last = null;   // to avoid loitering
            return item;
        }

       /**
         * Return string representation.
         */
        public String toString() {
            StringBuilder s = new StringBuilder();
            for (Item item : this)
                s.append(item + " ");
            return s.toString();
        }


       /**
         * Return an iterator that iterates over the items on the queue in FIFO order.
         */
        public Iterator<Item> iterator()  {
            return new FIFOIterator();
        }

        // an iterator, doesn't implement remove() since it's optional
        private class FIFOIterator implements Iterator<Item> {
            private Node current = first;

            public boolean hasNext()  { return current != null;                     }
            public void remove()      { throw new UnsupportedOperationException();  }

            public Item next() {
                if (!hasNext()) throw new NoSuchElementException();
                Item item = current.item;
                current = current.next;
                return item;
            }
        }
    }
}
